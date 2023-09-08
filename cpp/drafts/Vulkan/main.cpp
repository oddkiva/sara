#include <DO/Sara/Core/DebugUtilities.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Geometry.hpp"
#include "VulkanUtilities.hpp"

#include <set>


class SingleGLFWWindowApplication;
class VulkanBackend;


class SingleGLFWWindowApplication
{
  friend class VulkanBackend;

public:
  static constexpr auto default_width = 800;
  static constexpr auto default_height = 600;

  auto init() -> void
  {
    // Initialize the GLFW application.
    SARA_DEBUG << "[GLFW] Initializing the GLFW application...\n";
    const auto _glfw_app_initialized = glfwInit() == GLFW_TRUE;
    if (!_glfw_app_initialized)
      std::cerr << "Error: failed to initialize a GLFW application!\n";

    // Create a window.
    SARA_DEBUG << "[GLFW] Creating a single GLFW window...\n";
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    _window = glfwCreateWindow(default_width, default_height,  //
                               "Vulkan",                       //
                               nullptr, nullptr);
    if (_window == nullptr)
      std::cerr << "Error: failed to create a single GLFW window!\n";

    // Register C-style callbacks to this window.
    SARA_DEBUG << "[GLFW] Registering callbacks to the GLFW window...\n";
    glfwSetWindowUserPointer(_window, this);
    glfwSetFramebufferSizeCallback(_window, framebuffer_resize_callback);
  }

  auto cleanup() -> void
  {
    SARA_DEBUG << "[GLFW] Destroying the single GLFW window...\n";
    if (_window != nullptr)
      glfwDestroyWindow(_window);

    if (_glfw_app_initialized)
    {
      SARA_DEBUG << "[GLFW] Destroying the GLFW application...\n";
      glfwTerminate();
    }
  }

  static auto framebuffer_resize_callback(GLFWwindow* window,
                                          [[maybe_unused]] int width,
                                          [[maybe_unused]] int height) -> void
  {
    auto app = reinterpret_cast<SingleGLFWWindowApplication*>(  //
        glfwGetWindowUserPointer(window)                        //
    );

    // This is a dummy action.
    app->framebuffer_resized = true;
  }

private:
  bool _glfw_app_initialized = false;
  GLFWwindow* _window = nullptr;

  // TODO: window state.
  bool framebuffer_resized = false;
};


class VulkanBackend
{
public:
  //! For debugging purposes.
  static constexpr auto enable_validation_layers = true;
  static constexpr auto requested_validation_layers = std::array{
      "VK_LAYER_KHRONOS_validation"  //
  };

  //! Our Vulkan application must be able to support on-screen display
  //! operations.
  //!
  //! So we need to make sure our physical device supports the following
  //! extensions, namely only one extension, which is the swapchain.
  static constexpr auto required_device_extensions = std::array{
      VK_KHR_SWAPCHAIN_EXTENSION_NAME  //
  };

  //! Display control parameters.
  static constexpr auto max_frames_in_flight = 2;


  auto init(const SingleGLFWWindowApplication& app) -> void
  {
    init_instance("Hello Vulkan");
    init_debug_messengers();

    init_surface_from_glfw(app);

    find_first_suitable_vulkan_device();
    init_logical_device();

    init_swapchain(app);
  }

  auto cleanup() -> void
  {
    SARA_DEBUG << "[VK] Destroying the swapchain...\n";
    cleanup_swapchain();

    SARA_DEBUG << "[VK] Destroying the logical device...\n";
    vkDestroyDevice(_device, nullptr);

    // N.B.: the Vulkan physical device cannot be destroyed.

    SARA_DEBUG << "[VK] Destroying Vulkan surface...\n";
    vkDestroySurfaceKHR(_instance, _surface, nullptr);

    if (enable_validation_layers)
    {
      SARA_DEBUG << "[VK] Destroying debug utility messengers...\n";
      vk::destroy_debug_utils_messenger_ext(_instance, _debug_messenger,
                                            nullptr);
    }

    SARA_DEBUG << "[VK] Destroying Vulkan instance...\n";
    vkDestroyInstance(_instance, nullptr);
  }

private: /* initialization methods */
  auto init_instance(
      const std::string& application_name,
      const std::uint32_t application_version = VK_MAKE_VERSION(1, 0, 0),
      const std::string& engine_name = {},
      const std::uint32_t engine_version = VK_MAKE_VERSION(1, 0, 0)) -> bool
  {
    SARA_DEBUG << "[VK] Initializing a Vulkan instance...\n";
    if (enable_validation_layers &&
        !vk::check_validation_layer_support(requested_validation_layers))
    {
      SARA_DEBUG << "[VK] Error: requested Vulkan validation layers but they "
                    "are not available!\n";
      return false;
    }

    auto create_info = VkInstanceCreateInfo{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

    // 1. Fill in the application metadata that we - the developer - are making.
    auto app_info = VkApplicationInfo{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = application_name.c_str();
    app_info.applicationVersion = application_version;
    app_info.pEngineName = engine_name.c_str();
    app_info.engineVersion = engine_version;
    app_info.apiVersion = VK_API_VERSION_1_0;
    // Bind the application info object to the create info object.
    create_info.pApplicationInfo = &app_info;

    // 2. Fill in the application metadata that we - the developer - are making.
    const auto extensions = vk::get_required_extensions_from_glfw(  //
        enable_validation_layers                                    //
    );
    // Bind the array of GLFW extensions to the create info object.
    create_info.enabledExtensionCount = static_cast<uint32_t>(  //
        extensions.size()                                       //
    );
    create_info.ppEnabledExtensionNames = extensions.data();

    // 3. Specify the debug callback so that the Vulkan API produce debugging
    //    feedback.
    SARA_DEBUG << "[VK] Hooking debug callbacks to the Vulkan instance...\n";
    auto debug_create_info = VkDebugUtilsMessengerCreateInfoEXT{};
    vk::init_debug_messenger_create_info(debug_create_info);
    // Bind the debug create info to the create info object.
    if (enable_validation_layers)
    {
      create_info.enabledLayerCount = static_cast<uint32_t>(  //
          requested_validation_layers.size()                  //
      );
      create_info.ppEnabledLayerNames = requested_validation_layers.data();
      create_info.pNext = reinterpret_cast<VkDebugUtilsMessengerCreateInfoEXT*>(
          &debug_create_info);
    }
    else
    {
      create_info.enabledLayerCount = 0;
      create_info.pNext = nullptr;
    }

    // Finally instantiate the Vulkan instance.
    if (vkCreateInstance(&create_info, nullptr, &_instance) != VK_SUCCESS)
    {
      SARA_DEBUG << "failed to create instance!\n";
      return false;
    }

    return true;
  }

  auto init_debug_messengers() -> bool
  {
    VkDebugUtilsMessengerCreateInfoEXT create_info;
    vk::init_debug_messenger_create_info(create_info);

    if (vk::create_debug_utils_messenger_ext(_instance, &create_info, nullptr,
                                             &_debug_messenger) != VK_SUCCESS)
    {
      SARA_DEBUG << "[VK] failed to set up debug messengers!\n";
      return false;
    }

    return true;
  }

  auto init_surface_from_glfw(const SingleGLFWWindowApplication& app) -> bool
  {
    SARA_DEBUG
        << "[VK] Initializing Vulkan surface with the GLFW application...\n";
    if (glfwCreateWindowSurface(_instance,             //
                                app._window, nullptr,  //
                                &_surface) != VK_SUCCESS)
    {
      SARA_DEBUG << "[VK] Error: failed to initilialize Vulkan surface!\n";
      return false;
    }

    return true;
  }

  auto find_first_suitable_vulkan_device() -> bool
  {
    SARA_DEBUG << "[VK] Finding first suitable Vulkan physical device...\n";
    const auto physical_devices = vk::list_physical_devices(_instance);

    const auto physical_device_it = std::find_if(          //
        physical_devices.begin(), physical_devices.end(),  //
        [this](const auto& physical_device) {
          return physical_device_supports_required_extensions(
                     physical_device) &&
                 physical_device_supports_graphics_and_present_operations(
                     physical_device) &&
                 physical_device_supports_swapchain(physical_device);
        });
    if (physical_device_it == physical_devices.end())
    {
      std::cerr << "  [VK] Failed to find a suitable Vulkan physical device!\n";
      _physical_device = VK_NULL_HANDLE;
      return false;
    };

    _physical_device = *physical_device_it;
    return true;
  }

  // Create a logical device with:
  // - a graphics queue and
  // - a present queue.
  auto init_logical_device() -> bool
  {
    SARA_DEBUG << "[VK] Initializing the Vulkan logical device...\n";
    SARA_DEBUG
        << "[VK] - Querying queue families from the physical device...\n";
    const auto indices = find_queue_families(_physical_device);

    // Specify what the logical device that we want to create.
    //
    // Basically we must bind the capabilities of the physical device to this
    // logical device.
    auto create_info = VkDeviceCreateInfo{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    // 1. First we want the logical device to be able to run graphics and
    //    present operations.
    //
    //    So We need to bind these family of operations from the physical device
    //    to the logical device.
    SARA_DEBUG << "[VK] - Checking that the physical device supports graphics "
                  "and present operations...\n";
    auto queue_create_infos = std::vector<VkDeviceQueueCreateInfo>{};
    auto unique_queue_families = std::set<std::uint32_t>{
        indices.graphics_family.value(),  //
        indices.present_family.value()    //
    };

    static constexpr auto queue_priority = 1.f;
    for (const auto& queue_family : unique_queue_families)
    {
      auto queue_create_info = VkDeviceQueueCreateInfo{};
      queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queue_create_info.queueFamilyIndex = queue_family;
      queue_create_info.queueCount = 1;
      queue_create_info.pQueuePriorities = &queue_priority;
      queue_create_infos.emplace_back(queue_create_info);
    }
    create_info.queueCreateInfoCount =
        static_cast<std::uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();

    // 2. Bind the validation layers for debugging purposes.
    create_info.enabledLayerCount =
        enable_validation_layers
            ? static_cast<std::uint32_t>(requested_validation_layers.size())
            : std::uint32_t{};
    create_info.ppEnabledLayerNames =
        enable_validation_layers ? requested_validation_layers.data() : nullptr;

    // 3. Bind the Vulkan device extensions.
    create_info.enabledExtensionCount = static_cast<std::uint32_t>(  //
        required_device_extensions.size()                            //
    );
    create_info.ppEnabledExtensionNames = required_device_extensions.data();

    // 4. Bind any device features.
    static constexpr auto device_features = VkPhysicalDeviceFeatures{};
    create_info.pEnabledFeatures = &device_features;

    SARA_DEBUG << "[VK] - Initializing a Vulkan logical device from the Vulkan "
                  "physical device...\n";
    if (vkCreateDevice(_physical_device, &create_info, nullptr, &_device) !=
        VK_SUCCESS)
    {
      std::cerr << "[VK] Error: could not create a logical Vulkan device!\n";
      return false;
    }

    SARA_DEBUG
        << "[VK] - Fetching the graphics queue from the logical device...\n";
    vkGetDeviceQueue(_device, indices.graphics_family.value(), 0,
                     &_graphics_queue);
    SARA_DEBUG
        << "[VK] - Fetching the present queue from the logical device...\n";
    vkGetDeviceQueue(_device, indices.present_family.value(), 0,
                     &_present_queue);

    return true;
  }

  auto init_swapchain(const SingleGLFWWindowApplication& app) -> bool
  {
    const auto swapchain_support = vk::query_swapchain_support(  //
        _physical_device,                                        //
        _surface                                                 //
    );

    // Find a valid pixel format for the swap chain images and if possible
    // choose RGBA 32 bits.
    auto surface_format = choose_swap_surface_format(swapchain_support.formats);
    // FIFO/Mailbox presentation mode.
    auto present_mode = choose_swap_present_mode(  //
        swapchain_support.present_modes            //
    );
    // Set the swap chain image sizes to be equal to the window surface
    // sizes.
    const auto extent = choose_swap_extent(app, swapchain_support.capabilities);

    auto image_count = swapchain_support.capabilities.maxImageCount + 1;
    if (swapchain_support.capabilities.maxImageCount > 0 &&
        image_count > swapchain_support.capabilities.maxImageCount)
      image_count = swapchain_support.capabilities.maxImageCount;

    auto create_info = VkSwapchainCreateInfoKHR{};
    {
      create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
      create_info.surface = _surface;
      create_info.minImageCount = image_count;
      create_info.imageFormat = surface_format.format;
      create_info.imageColorSpace = surface_format.colorSpace;
      create_info.imageExtent = extent;
      create_info.imageArrayLayers = 1;
      create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }

    const auto indices = find_queue_families(_physical_device);
    const auto queue_family_indices = std::array{
        indices.graphics_family.value(),  //
        indices.present_family.value()    //
    };

    if (indices.graphics_family != indices.present_family)
    {
      create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      create_info.queueFamilyIndexCount = 2;
      create_info.pQueueFamilyIndices = queue_family_indices.data();
    }
    else
      create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

    create_info.preTransform = swapchain_support.capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(_device, &create_info, nullptr, &_swapchain) !=
        VK_SUCCESS)
    {
      std::cerr << "  [VK] Failed to create swapchain!\n";
      return false;
    }

    _swapchain_images = vk::list_swapchain_images(_device, _swapchain);
    _swapchain_image_format = surface_format.format;
    _swapchain_extent = extent;

    return true;
  }

  auto init_swapchain_image_views() -> bool
  {
    SARA_DEBUG << "[VK] Create swapchain image views...\n";
    _swapchain_image_views.resize(_swapchain_images.size());

    for (auto i = 0u; i < _swapchain_images.size(); ++i)
    {
      auto create_info = VkImageViewCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      create_info.image = _swapchain_images[i];
      create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      create_info.format = _swapchain_image_format;
      create_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY,  //
                                VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY};
      create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      create_info.subresourceRange.baseMipLevel = 0;
      create_info.subresourceRange.levelCount = 1;
      create_info.subresourceRange.baseArrayLayer = 0;
      create_info.subresourceRange.layerCount = 1;

      if (vkCreateImageView(_device, &create_info, nullptr,
                            &_swapchain_image_views[i]) != VK_SUCCESS)
      {
        std::cerr << "  [VK] Failed to create image views!\n";
        return false;
      }

      return true;
    }
  }

  // auto init_swapchain_framebuffers() -> bool
  // {
  //   _swapchain_framebuffers.resize(_swapchain_image_views.size());

  //   for (auto i = 0u; i < _swapchain_image_views.size(); ++i)
  //   {
  //     auto framebuffer_info = VkFramebufferCreateInfo{};
  //     {
  //       framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  //       framebuffer_info.renderPass = _render_pass;
  //       framebuffer_info.attachmentCount = 1;
  //       framebuffer_info.pAttachments = &_swapchain_image_views[i];
  //       framebuffer_info.width = _swapchain_extent.width;
  //       framebuffer_info.height = _swapchain_extent.height;
  //       framebuffer_info.layers = 1;
  //     }

  //     if (vkCreateFramebuffer(_device, &framebuffer_info, nullptr,
  //                             &_swapchain_framebuffers[i]) != VK_SUCCESS)
  //     {
  //       std::cerr << "  [VK] Failed to create framebuffer!\n";
  //       return false;
  //     }
  //   }

  //   return true;
  // }


private: /* cleanup methods related to the render pipeline. */
  // auto cleanup_render_pipeline() -> void {
  //   vkFreeCommandBuffers(_device, commandPool,
  //                        static_cast<uint32_t>(commandBuffers.size()),
  //                        commandBuffers.data());
  //   vkDestroyPipeline(_device, _graphics_pipeline, nullptr);
  //   vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
  //   vkDestroyRenderPass(_device, _render_pass, nullptr);
  // }

private: /* cleanup methods related to the present operations */
  auto cleanup_swapchain_framebuffers() -> void
  {
    for (const auto framebuffer : swapchain_framebuffers)
      vkDestroyFramebuffer(_device, framebuffer, nullptr);
  }

  void cleanup_swapchain_image_views()
  {
    for (const auto image_view : _swapchain_image_views)
      vkDestroyImageView(_device, image_view, nullptr);
  }

  void cleanup_swapchain()
  {
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);
  }

private: /* utility methods */
  // We need a GPU that supports Vulkan Graphics operations at the bare
  // minimum.
  auto find_queue_families(VkPhysicalDevice device) const
      -> vk::QueueFamilyIndices
  {
    auto indices = vk::QueueFamilyIndices{};

    auto queue_family_count = std::uint32_t{};
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             nullptr);

    auto queue_families =
        std::vector<VkQueueFamilyProperties>(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             queue_families.data());

    auto i = 0;
    for (const auto& queue_family : queue_families)
    {
      // Does the physical device have a graphics queue?
      if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        indices.graphics_family = i;

      // Does the physical device have a present queue?
      auto present_support = VkBool32{false};
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface,
                                           &present_support);
      if (present_support)
        indices.present_family = i;

      if (indices.is_complete())
        break;

      ++i;
    }

    return indices;
  }

  auto physical_device_supports_required_extensions(
      VkPhysicalDevice physical_device) const -> bool
  {
    const auto physical_device_extensions = vk::list_device_extensions(  //
        physical_device                                                  //
    );

    // Check that the required device extensions are supported by the physical
    // device.
    auto required_extensions_not_supported = std::set<std::string>(
        required_device_extensions.begin(), required_device_extensions.end());
    for (const auto& extension : physical_device_extensions)
      required_extensions_not_supported.erase(extension.extensionName);
    return required_extensions_not_supported.empty();
  }

  auto physical_device_supports_graphics_and_present_operations(
      VkPhysicalDevice physical_device) const -> bool
  {
    const auto indices = find_queue_families(physical_device);
    return indices.is_complete();
  }

  auto physical_device_supports_swapchain(
      const VkPhysicalDevice physical_device) const -> bool
  {
    const auto swapchain_support =
        vk::query_swapchain_support(physical_device, _surface);
    const auto swapchain_adequate = !swapchain_support.formats.empty() &&
                                    !swapchain_support.present_modes.empty();
    return swapchain_adequate;
  }

  //! We want a surface with the BGRA 32bit pixel format preferably, otherwise
  //! choose the first available surface format.
  auto choose_swap_surface_format(
      const std::vector<VkSurfaceFormatKHR>& available_formats) const
      -> VkSurfaceFormatKHR
  {
    for (const auto& available_format : available_formats)
    {
      if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
          available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        return available_format;
    }

    return available_formats[0];
  }

  // Choose the simplest (not the best one) swap present mode.
  auto choose_swap_present_mode(
      const std::vector<VkPresentModeKHR>& available_present_modes) const
      -> VkPresentModeKHR
  {
    for (const auto& available_present_mode : available_present_modes)
      if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
        return available_present_mode;

    // This is always available in any vulkan vendor implementation.
    // Preferred in mobile development.
    return VK_PRESENT_MODE_FIFO_KHR;
  }

  // Choose the swap image sizes.
  auto choose_swap_extent(const SingleGLFWWindowApplication& app,
                          const VkSurfaceCapabilitiesKHR& capabilities) const
      -> VkExtent2D
  {
    if (capabilities.currentExtent.width != UINT32_MAX)
      return capabilities.currentExtent;

    auto w = int{};
    auto h = int{};
    glfwGetFramebufferSize(app._window, &w, &h);

    auto actual_extent = VkExtent2D{
        .width = std::clamp(static_cast<std::uint32_t>(w),
                            capabilities.minImageExtent.width,
                            capabilities.maxImageExtent.width),
        .height = std::clamp(static_cast<std::uint32_t>(h),
                             capabilities.minImageExtent.height,
                             capabilities.maxImageExtent.height)  //
    };

    return actual_extent;
  }

private:
  // The Vulkan application.
  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;

  // The rendering surface.
  VkSurfaceKHR _surface;

  // The Vulkan-compatible GPU device.
  //
  // N.B.: no need to destroy this object.
  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;

  // The logical Vulkan device to which the physical device is bound.
  VkDevice _device;

  // The Vulkan capabilities that the logical device needs to have:
  // - Graphics rendering operations
  // - Display operations
  //
  // N.B.: no need to destroy these objects.
  VkQueue _graphics_queue;
  VkQueue _present_queue;

  // The swapchain.
  VkSwapchainKHR _swapchain;
  std::vector<VkImage> _swapchain_images;
  VkFormat _swapchain_image_format;
  VkExtent2D _swapchain_extent;

  std::vector<VkImageView> _swapchain_image_views;
  std::vector<VkFramebuffer> _swapchain_framebuffers;
};


class HelloTriangleApplication
{
public:
  void run()
  {
    init();
    cleanup();
  }

private:
  auto init() -> void
  {
    _glfw_window.init();
    _vulkan_backend.init(_glfw_window);
  }

  auto cleanup() -> void
  {
    _vulkan_backend.cleanup();
    _glfw_window.cleanup();
  }

private:
  SingleGLFWWindowApplication _glfw_window;
  VulkanBackend _vulkan_backend;
};


int main(int, char**)
{
  auto app = HelloTriangleApplication{};
  app.run();

  return 0;
}
