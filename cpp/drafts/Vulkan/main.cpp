#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <vector>


const auto WIDTH = 800;
const auto HEIGHT = 600;

const auto validation_layers = std::vector<const char*>{
    "VK_LAYER_KHRONOS_validation"  //
};

const auto device_extensions = std::vector<const char*>{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME  //
};

#ifdef NDEBUG
constexpr auto enable_validation_layers = false;
#else
constexpr auto enable_validation_layers = true;
#endif

namespace vk {

  auto CreateDebugUtilsMessengerExt(
      VkInstance instance, VkDebugUtilsMessengerCreateInfoEXT* createInfo,
      const VkAllocationCallbacks* allocator,
      VkDebugUtilsMessengerEXT* debugMessenger) -> VkResult
  {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
      return func(instance, createInfo, allocator, debugMessenger);
    else
      return VK_ERROR_EXTENSION_NOT_PRESENT;
  }

  auto DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                     VkDebugUtilsMessengerEXT debugMessenger,
                                     const VkAllocationCallbacks* allocator)
      -> void
  {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
      return func(instance, debugMessenger, allocator);
  }

  struct QueueFamilyIndices
  {
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;

    auto isComplete() const -> bool
    {
      return graphicsFamily.has_value() && presentFamily.has_value();
    }
  };

  struct SwapChainSupportDetails
  {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
  };

}  // namespace vk


class HelloTriangle
{
public:
  void run()
  {
    init_window();
    init_vulkan();
    main_loop();
    cleanup();
  }

private:
  auto init_window() -> void
  {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RELEASE, GLFW_FALSE);

    _window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

  void init_vulkan()
  {
    create_vulkan_instance();
    setup_debug_messenger();
    create_surface();
    pick_physical_device();
    create_logical_device();
    create_swapchain();
    create_image_views();
    create_graphics_pipeline();
  }

  void main_loop()
  {
    while (!glfwWindowShouldClose(_window))
      glfwPollEvents();
  }

  void cleanup()
  {
    if (enable_validation_layers)
      vk::DestroyDebugUtilsMessengerEXT(_instance, _debug_messenger, nullptr);

    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkDestroyInstance(_instance, nullptr);

    glfwDestroyWindow(_window);
    glfwTerminate();
  }

  void create_vulkan_instance()
  {
    if (enable_validation_layers && !check_validation_layer_support())
      throw std::runtime_error{
          "Validation layers requested but not available!"  //
      };

    auto app_info = VkApplicationInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0  //
    };

    auto create_info = VkInstanceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,  //
        .pApplicationInfo = &app_info,
    };

    auto extensions = get_required_extensions();
    create_info.enabledExtensionCount =
        static_cast<std::uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    auto debug_create_info = VkDebugUtilsMessengerCreateInfoEXT{};
    if (enable_validation_layers)
    {
      create_info.enabledLayerCount =
          static_cast<std::uint32_t>(validation_layers.size());
      create_info.ppEnabledLayerNames = validation_layers.data();

      populate_debug_messenger_create_info(debug_create_info);
      create_info.pNext =
          (VkDebugUtilsMessengerCreateInfoEXT*) &debug_create_info;
    }
    else
    {
      create_info.enabledLayerCount = 0;
      create_info.pNext = nullptr;
    }

    if (vkCreateInstance(&create_info, nullptr, &_instance) != VK_SUCCESS)
      throw std::runtime_error{"Failed to create instance!"};
  }

  auto populate_debug_messenger_create_info(
      VkDebugUtilsMessengerCreateInfoEXT& create_info) const -> void
  {
    create_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |  //
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |  //
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_callback};
  }

  auto setup_debug_messenger() -> void
  {
    if (!enable_validation_layers)
      return;

    auto create_info = VkDebugUtilsMessengerCreateInfoEXT{};
    populate_debug_messenger_create_info(create_info);

    if (vk::CreateDebugUtilsMessengerExt(_instance, &create_info, nullptr,
                                         &_debug_messenger) != VK_SUCCESS)
      throw std::runtime_error{"Failed to set up debug messenger"};
  }

  auto get_required_extensions() const -> std::vector<const char*>
  {
    auto glfw_extension_count = std::uint32_t{};
    const char** glfw_extensions = nullptr;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    auto extensions = std::vector<const char*>(  //
        glfw_extensions,                         //
        glfw_extensions + glfw_extension_count   //
    );

    if (enable_validation_layers)
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
  }

  auto check_validation_layer_support() const -> bool
  {
    auto layer_count = std::uint32_t{};
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    auto available_layers = std::vector<VkLayerProperties>(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    for (const auto& layerName : validation_layers)
    {
      auto layer_found = false;

      for (const auto& layer_properties : available_layers)
      {
        if (strcmp(layerName, layer_properties.layerName) == 0)
        {
          layer_found = true;
          break;
        }
      }

      if (!layer_found)
        return false;
    }

    return true;
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
      VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
      const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void*)
  {
    std::cerr << "Validation layer:" << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
  }

  auto pick_physical_device() -> void
  {
    auto device_count = std::uint32_t{};
    vkEnumeratePhysicalDevices(_instance, &device_count, nullptr);

    if (device_count == 0)
      throw std::runtime_error{"Failed to find GPUs with Vulkan support"};

    auto devices = std::vector<VkPhysicalDevice>(device_count);
    vkEnumeratePhysicalDevices(_instance, &device_count, devices.data());

    for (const auto& device : devices)
    {
      if (is_device_suitable(device))
      {
        _physical_device = device;
        break;
      }
    }

    if (_physical_device == VK_NULL_HANDLE)
      throw std::runtime_error{"Failed to find a suitable GPU!"};
  }

  auto is_device_suitable(VkPhysicalDevice device) const -> bool
  {
    const auto indices = find_queue_families(device);

    const auto extensions_supported = check_device_extension_support(device);
    auto swapchain_adequate = false;

    if (extensions_supported)
    {
      auto swapchain_support = query_swapchain_support(device);
      swapchain_adequate = !swapchain_support.formats.empty() &&
                           !swapchain_support.presentModes.empty();
    }

    return indices.isComplete() && extensions_supported && swapchain_adequate;
  }

  auto check_device_extension_support(VkPhysicalDevice device) const -> bool
  {
    auto extension_count = std::uint32_t{};
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                         nullptr);

    auto available_extensions =
        std::vector<VkExtensionProperties>(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                         available_extensions.data());

    auto requiredExtensions = std::set<std::string>(device_extensions.begin(),
                                                    device_extensions.end());

    for (const auto& extension : available_extensions)
      requiredExtensions.erase(extension.extensionName);

    return requiredExtensions.empty();
  }

  // We need a GPU that supports Vulkan Graphics operations at the bare minimum.
  auto find_queue_families(VkPhysicalDevice device) const
      -> vk::QueueFamilyIndices
  {
    auto indices = vk::QueueFamilyIndices{};

    auto queue_family_count = std::uint32_t{};
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             nullptr);

    auto queue_families = std::vector<VkQueueFamilyProperties>(  //
        queue_family_count                                       //
    );
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             queue_families.data());

    auto i = 0;
    for (const auto& queue_family : queue_families)
    {
      // Does the physical device have a graphics queue?
      if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        indices.graphicsFamily = i;

      // Does the physical device have a present queue?
      auto present_support = VkBool32{false};
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface,
                                           &present_support);
      if (present_support)
        indices.presentFamily = i;

      if (indices.isComplete())
        break;

      ++i;
    }

    return indices;
  }

  // Create a logical device with:
  // - a graphics queue and
  // - a present queue.
  auto create_logical_device() -> void
  {
    const auto indices = find_queue_families(_physical_device);

    auto queue_create_infos = std::vector<VkDeviceQueueCreateInfo>{};
    auto unique_queue_families = std::set<std::uint32_t>{
        indices.graphicsFamily.value(),  //
        indices.presentFamily.value()    //
    };

    auto queue_priority = 1.f;

    for (const auto& queue_family : unique_queue_families)
    {
      auto queue_create_info = VkDeviceQueueCreateInfo{
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = queue_family,
          .queueCount = 1,
          .pQueuePriorities = &queue_priority  //
      };
      queue_create_infos.emplace_back(queue_create_info);
    }

    auto device_features = VkPhysicalDeviceFeatures{};

    auto create_info = VkDeviceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount =
            static_cast<std::uint32_t>(queue_create_infos.size()),
        .pQueueCreateInfos = queue_create_infos.data(),
        .enabledLayerCount =
            enable_validation_layers
                ? static_cast<std::uint32_t>(validation_layers.size())
                : 0u,
        .ppEnabledLayerNames =
            enable_validation_layers ? validation_layers.data() : nullptr,
        .enabledExtensionCount =
            static_cast<std::uint32_t>(device_extensions.size()),
        .ppEnabledExtensionNames = device_extensions.data(),
        .pEnabledFeatures = &device_features  //
    };

    if (vkCreateDevice(_physical_device, &create_info, nullptr, &_device) !=
        VK_SUCCESS)
      throw std::runtime_error{"Failed to create logical device!"};

    vkGetDeviceQueue(_device, indices.graphicsFamily.value(), 0,
                     &_graphics_queue);
    vkGetDeviceQueue(_device, indices.presentFamily.value(), 0,
                     &_present_queue);
  }

  auto create_surface() -> void
  {
    if (glfwCreateWindowSurface(_instance, _window, nullptr, &_surface) !=
        VK_SUCCESS)
      throw std::runtime_error{"Failed to create window surface!"};
  }

  auto create_swapchain() -> void
  {
    auto swapchain_support = query_swapchain_support(_physical_device);

    auto surface_format = choose_swap_surface_format(swapchain_support.formats);
    auto present_mode = choose_swap_present_mode(  //
        swapchain_support.presentModes             //
    );
    auto extent = choose_swap_extent(swapchain_support.capabilities);

    auto image_count = swapchain_support.capabilities.maxImageCount + 1;
    if (swapchain_support.capabilities.maxImageCount > 0 &&
        image_count > swapchain_support.capabilities.maxImageCount)
      image_count = swapchain_support.capabilities.maxImageCount;

    auto create_info = VkSwapchainCreateInfoKHR{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = _surface,
        .minImageCount = image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT  //
    };

    auto indices = find_queue_families(_physical_device);
    auto queue_family_indices = std::array{
        indices.graphicsFamily.value(),  //
        indices.presentFamily.value()    //
    };

    if (indices.graphicsFamily != indices.presentFamily)
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
      throw std::runtime_error{"Failed to create swap chain!"};

    vkGetSwapchainImagesKHR(_device, _swapchain, &image_count, nullptr);
    _swapchain_images.resize(image_count);
    vkGetSwapchainImagesKHR(_device, _swapchain, &image_count,
                            _swapchain_images.data());

    _swapchain_image_format = surface_format.format;
    _swapchain_extent = extent;
  }

  auto query_swapchain_support(VkPhysicalDevice device) const
      -> vk::SwapChainSupportDetails
  {
    auto details = vk::SwapChainSupportDetails{};

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, _surface,
                                              &details.capabilities);

    auto format_count = std::uint32_t{};
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &format_count,
                                         nullptr);
    if (format_count != 0)
    {
      details.formats.resize(format_count);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &format_count,
                                           details.formats.data());
    }

    auto present_mode_count = std::uint32_t{};
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface,
                                              &present_mode_count, nullptr);
    if (present_mode_count != 0)
    {
      details.presentModes.resize(present_mode_count);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          device, _surface, &present_mode_count, details.presentModes.data());
    }

    return details;
  }

  // We want a surface with the BGRA 32bit pixel format preferably.
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

    return VK_PRESENT_MODE_FIFO_KHR;
  }

  // Choose the swap image sizes.
  auto choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities) const
      -> VkExtent2D
  {
    if (capabilities.currentExtent.width != UINT32_MAX)
      return capabilities.currentExtent;

    auto w = int{};
    auto h = int{};
    glfwGetFramebufferSize(_window, &w, &h);

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

  auto create_image_views() -> void
  {
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
        throw std::runtime_error{"Failed to create image views!"};
    }
  }

  auto create_graphics_pipeline() -> void
  {
    auto vertex_shader_code = read_shader_file("vert.spv");
    auto fragment_shader_code = read_shader_file("frag.spv");

    auto vertex_shader_module = create_shader_module(vertex_shader_code);
    auto fragment_shader_module = create_shader_module(fragment_shader_code);

    auto vertex_shader_stage_info = VkPipelineShaderStageCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vertex_shader_module,
      .pName = "main"
    };

    auto fragment_shader_stage_info = VkPipelineShaderStageCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = fragment_shader_module,
      .pName = "main"
    };

    auto shader_stages = std::array{
      vertex_shader_stage_info, fragment_shader_stage_info
    };

    vkDestroyShaderModule(_device, vertex_shader_module, nullptr);
    vkDestroyShaderModule(_device, fragment_shader_module, nullptr);
  }

  static auto read_shader_file(const std::string& filename) -> std::vector<char>
  {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file)
      throw std::runtime_error{"Failed to open file!"};

    const auto file_size = static_cast<std::size_t>(file.tellg());
    auto buffer = std::vector<char>(file_size);
    std::cout << "buffer.size() = " << buffer.size()<< std::endl;

    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();

    return buffer;
  }

  auto create_shader_module(const std::vector<char>& buffer)
      -> VkShaderModule
  {
    auto create_info = VkShaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = static_cast<std::uint32_t>(buffer.size()),
        .pCode = reinterpret_cast<const std::uint32_t *>(buffer.data())};

    auto shader_module = VkShaderModule{};
    if (vkCreateShaderModule(_device, &create_info, nullptr, &shader_module) !=
        VK_SUCCESS)
      throw std::runtime_error{"Failed to create shader module!"};

    return shader_module;
  }

  GLFWwindow* _window = nullptr;

  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;
  VkSurfaceKHR _surface;

  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
  VkDevice _device;

  VkQueue _graphics_queue;
  VkQueue _present_queue;

  VkSwapchainKHR _swapchain;
  std::vector<VkImage> _swapchain_images;
  VkFormat _swapchain_image_format;
  VkExtent2D _swapchain_extent;

  std::vector<VkImageView> _swapchain_image_views;
};


int main()
{
  HelloTriangle app;

  try
  {
    app.run();
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
