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
  static constexpr auto device_extensions = std::array{
      VK_KHR_SWAPCHAIN_EXTENSION_NAME  //
  };

  //! Display control parameters.
  static constexpr auto max_frames_in_flight = 2;


  auto init(const SingleGLFWWindowApplication& app) -> void
  {
    init_instance("Hello Vulkan");
    init_debug_messengers();
    // if (enable_validation_layers)
    //   init_surface_from_glfw(app);
  }

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

  auto init_surface_from_glfw(const SingleGLFWWindowApplication& app) -> bool
  {
    if (glfwCreateWindowSurface(_instance,             //
                                app._window, nullptr,  //
                                &_surface) != VK_SUCCESS)
    {
      SARA_DEBUG << "[VK] Error: failed to initilialize Vulkan surface!\n";
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

  // Create a logical device with:
  // - a graphics queue and
  // - a present queue.
  auto init_logical_device() -> void
  {
    const auto indices = find_queue_families(_physical_device);

    auto queue_create_infos = std::vector<VkDeviceQueueCreateInfo>{};
    auto unique_queue_families = std::set<std::uint32_t>{
        indices.graphics_family.value(),  //
        indices.present_family.value()    //
    };

    auto queue_priority = 1.f;

    for (const auto& queue_family : unique_queue_families)
    {
      auto queue_create_info = VkDeviceQueueCreateInfo{};
      queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queue_create_info.queueFamilyIndex = queue_family;
      queue_create_info.queueCount = 1;
      queue_create_info.pQueuePriorities = &queue_priority;
      queue_create_infos.emplace_back(queue_create_info);
    }

    auto device_features = VkPhysicalDeviceFeatures{};

    auto create_info = VkDeviceCreateInfo{};
    {
      create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      create_info.queueCreateInfoCount =
          static_cast<std::uint32_t>(queue_create_infos.size());
      create_info.pQueueCreateInfos = queue_create_infos.data();
      create_info.enabledLayerCount =
          enable_validation_layers
              ? static_cast<std::uint32_t>(requested_validation_layers.size())
              : std::uint32_t{};
      create_info.ppEnabledLayerNames = enable_validation_layers
                                            ? requested_validation_layers.data()
                                            : nullptr;
      create_info.enabledExtensionCount =
          static_cast<std::uint32_t>(device_extensions.size());
      create_info.ppEnabledExtensionNames = device_extensions.data();
      create_info.pEnabledFeatures = &device_features;
    };

    if (vkCreateDevice(_physical_device, &create_info, nullptr, &_device) !=
        VK_SUCCESS)
      throw std::runtime_error{"Failed to create logical device!"};

    vkGetDeviceQueue(_device, indices.graphics_family.value(), 0,
                     &_graphics_queue);
    vkGetDeviceQueue(_device, indices.present_family.value(), 0,
                     &_present_queue);
  }

  auto cleanup() -> void
  {
    if (enable_validation_layers)
    {
      SARA_DEBUG << "[VK] Cleaning debug utility messengers...\n";
      vk::destroy_debug_utils_messenger_ext(_instance, _debug_messenger,
                                            nullptr);
    }

    // vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkDestroyInstance(_instance, nullptr);
  }

private:
  auto find_first_vulkan_compatible_physical_device() -> void
  {
    auto device_count = std::uint32_t{};
    vkEnumeratePhysicalDevices(_instance, &device_count, nullptr);

    if (device_count == 0)
      throw std::runtime_error("failed to find GPUs with Vulkan support!");

    auto devices = std::vector<VkPhysicalDevice>(device_count);
    vkEnumeratePhysicalDevices(_instance, &device_count, devices.data());

    const auto physical_device_it = std::find_if(  //
        devices.begin(), devices.end(),            //
        [this](const auto& device) {
          return physical_device_is_vulkan_compatible(device);
        });
    if (physical_device_it == devices.end())
      throw std::runtime_error{"Failed to find a suitable GPU!"};

    _physical_device = *physical_device_it;
  }

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

  auto physical_device_is_vulkan_compatible(VkPhysicalDevice device) const
      -> bool
  {
    const auto indices = find_queue_families(device);

    const auto extensions_supported =
        physical_device_supports_vulkan_extensions(device);
    if (!extensions_supported)
      return false;

    const auto swapchain_support = query_swapchain_support(device);
    const auto swapchain_adequate = !swapchain_support.formats.empty() &&
                                    !swapchain_support.present_modes.empty();

    return indices.is_complete() && extensions_supported && swapchain_adequate;
  }

  auto physical_device_supports_vulkan_extensions(VkPhysicalDevice device) const
      -> bool
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
      details.present_modes.resize(present_mode_count);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          device, _surface, &present_mode_count, details.present_modes.data());
    }

    return details;
  }


private:
  // The Vulkan application.
  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;

  // The rendering surface.
  VkSurfaceKHR _surface;

  // The Vulkan-compatible GPU device.
  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;

  VkDevice _device;
  VkQueue _graphics_queue;
  VkQueue _present_queue;
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
