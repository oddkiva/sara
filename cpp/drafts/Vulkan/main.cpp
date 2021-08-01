// #include <vulkan/vulkan.h>
//
// #define GLFW_INCLUDE_VULKAN
// #include <GLFW/glfw3.h>
//
// #include <boost/filesystem.hpp>
//
// #include <Eigen/Core>
//
// #include <algorithm>
// #include <array>
// #include <cstdint>
// #include <cstdlib>
// #include <cstring>
// #include <fstream>
// #include <iostream>
// #include <optional>
// #include <set>
// #include <stdexcept>
// #include <vector>
//
// namespace fs = boost::filesystem;
//
// const uint32_t WIDTH = 800;
// const uint32_t HEIGHT = 600;
//
// const int MAX_FRAMES_IN_FLIGHT = 2;
//
// const std::vector<const char*> validation_layers = {
//   "VK_LAYER_KHRONOS_validation"};
//
// const std::vector<const char*> deviceExtensions = {
//   VK_KHR_SWAPCHAIN_EXTENSION_NAME};
//
// #ifdef NDEBUG
// const bool enableValidationLayers = false;
// #else
// const bool enable_validation_layers = true;
// #endif
//
// namespace vk {
// VkResult CreateDebugUtilsMessengerEXT(
//     VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
//     const VkAllocationCallbacks* pAllocator,
//     VkDebugUtilsMessengerEXT* pDebugMessenger)
// {
//   auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
//       instance, "vkCreateDebugUtilsMessengerEXT");
//   if (func != nullptr)
//   {
//     return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
//   }
//   else
//   {
//     return VK_ERROR_EXTENSION_NOT_PRESENT;
//   }
// }
//
// void DestroyDebugUtilsMessengerEXT(VkInstance instance,
//                                    VkDebugUtilsMessengerEXT debugMessenger,
//                                    const VkAllocationCallbacks* pAllocator)
// {
//   auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
//       instance, "vkDestroyDebugUtilsMessengerEXT");
//   if (func != nullptr)
//   {
//     func(instance, debugMessenger, pAllocator);
//   }
// }
// }
//
// struct QueueFamilyIndices
// {
//   std::optional<uint32_t> graphicsFamily;
//   std::optional<uint32_t> presentFamily;
//
//   bool isComplete()
//   {
//     return graphicsFamily.has_value() && presentFamily.has_value();
//   }
// };
//
// struct SwapChainSupportDetails
// {
//   VkSurfaceCapabilitiesKHR capabilities;
//   std::vector<VkSurfaceFormatKHR> formats;
//   std::vector<VkPresentModeKHR> presentModes;
// };
//
// struct Vertex
// {
//   Eigen::Vector2f pos;
//   Eigen::Vector3f color;
//
//   static VkVertexInputBindingDescription getBindingDescription()
//   {
//     VkVertexInputBindingDescription bindingDescription{};
//     bindingDescription.binding = 0;
//     bindingDescription.stride = sizeof(Vertex);
//     bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
//
//     return bindingDescription;
//   }
//
//   static std::array<VkVertexInputAttributeDescription, 2>
//     getAttributeDescriptions()
//     {
//       std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
//
//       attributeDescriptions[0].binding = 0;
//       attributeDescriptions[0].location = 0;
//       attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
//       attributeDescriptions[0].offset = offsetof(Vertex, pos);
//
//       attributeDescriptions[1].binding = 0;
//       attributeDescriptions[1].location = 1;
//       attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
//       attributeDescriptions[1].offset = offsetof(Vertex, color);
//
//       return attributeDescriptions;
//     }
// };
//
// const std::vector<Vertex> vertices = {{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
//   {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
//   {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};
//
// class HelloTriangleApplication
// {
// public:
//   std::string _program_path;
//
//   HelloTriangleApplication(const std::string& program_path)
//     : _program_path{program_path}
//   {
//   }
//
//   void run()
//   {
//     initWindow();
//     initVulkan();
//     mainLoop();
//     cleanup();
//   }
//
// private:
//   GLFWwindow* window;
//
//   VkInstance _instance;
//   VkDebugUtilsMessengerEXT _debug_messenger;
//   VkSurfaceKHR _surface;
//
//   VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
//   VkDevice _device;
//
//   VkQueue graphicsQueue;
//   VkQueue presentQueue;
//
//   VkSwapchainKHR swapChain;
//   std::vector<VkImage> swapChainImages;
//   VkFormat swapChainImageFormat;
//   VkExtent2D swapChainExtent;
//   std::vector<VkImageView> swapChainImageViews;
//   std::vector<VkFramebuffer> swapChainFramebuffers;
//
//   VkRenderPass _render_pass;
//   VkPipelineLayout _pipeline_layout;
//   VkPipeline _graphics_pipeline;
//
//   VkCommandPool commandPool;
//
//   VkBuffer vertexBuffer;
//   VkDeviceMemory vertexBufferMemory;
//
//   std::vector<VkCommandBuffer> commandBuffers;
//
//   std::vector<VkSemaphore> imageAvailableSemaphores;
//   std::vector<VkSemaphore> renderFinishedSemaphores;
//   std::vector<VkFence> inFlightFences;
//   std::vector<VkFence> imagesInFlight;
//   size_t currentFrame = 0;
//
//   bool framebufferResized = false;
//
//   void initWindow()
//   {
//     glfwInit();
//
//     glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//
//     window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
//     glfwSetWindowUserPointer(window, this);
//     glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
//   }
//
//   static void framebufferResizeCallback(GLFWwindow* window,
//                                         [[maybe_unused]] int width,
//                                         [[maybe_unused]] int height)
//   {
//     auto app = reinterpret_cast<HelloTriangleApplication*>(
//         glfwGetWindowUserPointer(window));
//     app->framebufferResized = true;
//   }
//
//   void initVulkan()
//   {
//     create_instance();
//     setup_debug_messenger();
//     createSurface();
//     pickPhysicalDevice();
//     createLogicalDevice();
//     createSwapChain();
//     createImageViews();
//     createRenderPass();
//     createGraphicsPipeline();
//     createFramebuffers();
//     createCommandPool();
//     createVertexBuffer();
//     createCommandBuffers();
//     createSyncObjects();
//   }
//
//   void mainLoop()
//   {
//     while (!glfwWindowShouldClose(window))
//     {
//       glfwPollEvents();
//       drawFrame();
//     }
//
//     vkDeviceWaitIdle(_device);
//   }
//
//   void cleanup_swap_chain()
//   {
//     for (auto framebuffer : swapChainFramebuffers)
//     {
//       vkDestroyFramebuffer(_device, framebuffer, nullptr);
//     }
//
//     vkFreeCommandBuffers(_device, commandPool,
//                          static_cast<uint32_t>(commandBuffers.size()),
//                          commandBuffers.data());
//
//     vkDestroyPipeline(_device, _graphics_pipeline, nullptr);
//     vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
//     vkDestroyRenderPass(_device, _render_pass, nullptr);
//
//     for (auto imageView : swapChainImageViews)
//     {
//       vkDestroyImageView(_device, imageView, nullptr);
//     }
//
//     vkDestroySwapchainKHR(_device, swapChain, nullptr);
//   }
//
//   void cleanup()
//   {
//     cleanup_swap_chain();
//
//     vkDestroyBuffer(_device, vertexBuffer, nullptr);
//     vkFreeMemory(_device, vertexBufferMemory, nullptr);
//
//     for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
//     {
//       vkDestroySemaphore(_device, renderFinishedSemaphores[i], nullptr);
//       vkDestroySemaphore(_device, imageAvailableSemaphores[i], nullptr);
//       vkDestroyFence(_device, inFlightFences[i], nullptr);
//     }
//
//     vkDestroyCommandPool(_device, commandPool, nullptr);
//
//     vkDestroyDevice(_device, nullptr);
//
//     if (enable_validation_layers)
//     {
//       DestroyDebugUtilsMessengerEXT(_instance, _debug_messenger, nullptr);
//     }
//
//     vkDestroySurfaceKHR(_instance, _surface, nullptr);
//     vkDestroyInstance(_instance, nullptr);
//
//     glfwDestroyWindow(window);
//
//     glfwTerminate();
//   }
//
//   void recreateSwapChain()
//   {
//     int width = 0, height = 0;
//     glfwGetFramebufferSize(window, &width, &height);
//     while (width == 0 || height == 0)
//     {
//       glfwGetFramebufferSize(window, &width, &height);
//       glfwWaitEvents();
//     }
//
//     vkDeviceWaitIdle(_device);
//
//     cleanup_swap_chain();
//
//     createSwapChain();
//     createImageViews();
//     createRenderPass();
//     createGraphicsPipeline();
//     createFramebuffers();
//     createCommandBuffers();
//
//     imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
//   }
//
//   void create_instance()
//   {
//     if (enable_validation_layers && !checkValidationLayerSupport())
//     {
//       throw std::runtime_error(
//           "validation layers requested, but not available!");
//     }
//
//     VkApplicationInfo appInfo{};
//     appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
//     appInfo.pApplicationName = "Hello Triangle";
//     appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
//     appInfo.pEngineName = "No Engine";
//     appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
//     appInfo.apiVersion = VK_API_VERSION_1_0;
//
//     VkInstanceCreateInfo createInfo{};
//     createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
//     createInfo.pApplicationInfo = &appInfo;
//
//     auto extensions = ();
//     createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
//     createInfo.ppEnabledExtensionNames = extensions.data();
//
//     VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
//     if (enable_validation_layers)
//     {
//       createInfo.enabledLayerCount =
//         static_cast<uint32_t>(validation_layers.size());
//       createInfo.ppEnabledLayerNames = validation_layers.data();
//
//       populateDebugMessengerCreateInfo(debugCreateInfo);
//       createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
//     }
//     else
//     {
//       createInfo.enabledLayerCount = 0;
//
//       createInfo.pNext = nullptr;
//     }
//
//     if (vkCreateInstance(&createInfo, nullptr, &_instance) != VK_SUCCESS)
//     {
//       throw std::runtime_error("failed to create instance!");
//     }
//   }
//
//   void populateDebugMessengerCreateInfo(
//       VkDebugUtilsMessengerCreateInfoEXT& createInfo)
//   {
//     createInfo = {};
//     createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
//     createInfo.messageSeverity =
//       VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
//       VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
//       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
//     createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
//       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
//       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
//     createInfo.pfnUserCallback = debugCallback;
//   }
//
//   void setup_debug_messenger()
//   {
//     if (!enable_validation_layers)
//       return;
//
//     VkDebugUtilsMessengerCreateInfoEXT createInfo;
//     populateDebugMessengerCreateInfo(createInfo);
//
//     if (CreateDebugUtilsMessengerEXT(_instance, &createInfo, nullptr,
//                                      &_debug_messenger) != VK_SUCCESS)
//     {
//       throw std::runtime_error("failed to set up debug messenger!");
//     }
//   }
//
//   void createSurface()
//   {
//     if (glfwCreateWindowSurface(_instance, window, nullptr, &_surface) !=
//         VK_SUCCESS)
//     {
//       throw std::runtime_error("failed to create window surface!");
//     }
//   }
//
//   void pickPhysicalDevice()
//   {
//     uint32_t deviceCount = 0;
//     vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr);
//
//     if (deviceCount == 0)
//     {
//       throw std::runtime_error("failed to find GPUs with Vulkan support!");
//     }
//
//     std::vector<VkPhysicalDevice> devices(deviceCount);
//     vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data());
//
//     for (const auto& device : devices)
//     {
//       if (isDeviceSuitable(device))
//       {
//         _physical_device = device;
//         break;
//       }
//     }
//
//     if (_physical_device == VK_NULL_HANDLE)
//       throw std::runtime_error{"Failed to find a suitable GPU!"};
//   }
//
//   auto is_device_suitable(VkPhysicalDevice device) const -> bool
//   {
//     const auto indices = find_queue_families(device);
//
//     const auto extensions_supported = check_device_extension_support(device);
//     auto swapchain_adequate = false;
//
//     if (extensions_supported)
//     {
//       auto swapchain_support = query_swapchain_support(device);
//       swapchain_adequate = !swapchain_support.formats.empty() &&
//                            !swapchain_support.presentModes.empty();
//     }
//
//     return indices.isComplete() && extensions_supported && swapchain_adequate;
//   }
//
//   auto check_device_extension_support(VkPhysicalDevice device) const -> bool
//   {
//     auto extension_count = std::uint32_t{};
//     vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
//                                          nullptr);
//
//     auto available_extensions =
//         std::vector<VkExtensionProperties>(extension_count);
//     vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
//                                          available_extensions.data());
//
//     auto requiredExtensions = std::set<std::string>(device_extensions.begin(),
//                                                     device_extensions.end());
//
//     for (const auto& extension : available_extensions)
//       requiredExtensions.erase(extension.extensionName);
//
//     return requiredExtensions.empty();
//   }
//
//   // We need a GPU that supports Vulkan Graphics operations at the bare minimum.
//   auto find_queue_families(VkPhysicalDevice device) const
//       -> vk::QueueFamilyIndices
//   {
//     auto indices = vk::QueueFamilyIndices{};
//
//     auto queue_family_count = std::uint32_t{};
//     vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
//                                              nullptr);
//
//     auto queue_families = std::vector<VkQueueFamilyProperties>(  //
//         queue_family_count                                       //
//     );
//     vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
//                                              queue_families.data());
//
//     auto i = 0;
//     for (const auto& queue_family : queue_families)
//     {
//       // Does the physical device have a graphics queue?
//       if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
//         indices.graphicsFamily = i;
//
//       // Does the physical device have a present queue?
//       auto present_support = VkBool32{false};
//       vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface,
//                                            &present_support);
//       if (present_support)
//         indices.presentFamily = i;
//
//       if (indices.isComplete())
//         break;
//
//       ++i;
//     }
//
//     return indices;
//   }
//
//   // Create a logical device with:
//   // - a graphics queue and
//   // - a present queue.
//   auto create_logical_device() -> void
//   {
//     const auto indices = find_queue_families(_physical_device);
//
//     auto queue_create_infos = std::vector<VkDeviceQueueCreateInfo>{};
//     auto unique_queue_families = std::set<std::uint32_t>{
//         indices.graphicsFamily.value(),  //
//         indices.presentFamily.value()    //
//     };
//
//     auto queue_priority = 1.f;
//
//     for (const auto& queue_family : unique_queue_families)
//     {
//       auto queue_create_info = VkDeviceQueueCreateInfo{};
//       queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
//       queue_create_info.queueFamilyIndex = queue_family;
//       queue_create_info.queueCount = 1;
//       queue_create_info.pQueuePriorities = &queue_priority;
//       queue_create_infos.emplace_back(queue_create_info);
//     }
//
//     auto device_features = VkPhysicalDeviceFeatures{};
//
//     auto create_info = VkDeviceCreateInfo{};
//     {
//       create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
//       create_info.queueCreateInfoCount =
//           static_cast<std::uint32_t>(queue_create_infos.size());
//       create_info.pQueueCreateInfos = queue_create_infos.data();
//       create_info.enabledLayerCount =
//           enable_validation_layers
//               ? static_cast<std::uint32_t>(validation_layers.size())
//               : 0u;
//       create_info.ppEnabledLayerNames =
//           enable_validation_layers ? validation_layers.data() : nullptr;
//       create_info.enabledExtensionCount =
//           static_cast<std::uint32_t>(device_extensions.size());
//       create_info.ppEnabledExtensionNames = device_extensions.data();
//       create_info.pEnabledFeatures = &device_features;
//     };
//
//     if (vkCreateDevice(_physical_device, &create_info, nullptr, &_device) !=
//         VK_SUCCESS)
//       throw std::runtime_error{"Failed to create logical device!"};
//
//     vkGetDeviceQueue(_device, indices.graphicsFamily.value(), 0,
//                      &_graphics_queue);
//     vkGetDeviceQueue(_device, indices.presentFamily.value(), 0,
//                      &_present_queue);
//   }
//
//   auto create_surface() -> void
//   {
//     if (glfwCreateWindowSurface(_instance, _window, nullptr, &_surface) !=
//         VK_SUCCESS)
//       throw std::runtime_error{"Failed to create window surface!"};
//   }
//
//   auto create_swapchain() -> void
//   {
//     auto swapchain_support = query_swapchain_support(_physical_device);
//
//     // Find a valid pixel format for the swap chain images and if possible
//     // choose RGBA 32 bits.
//     auto surface_format = choose_swap_surface_format(swapchain_support.formats);
//     // FIFO/Mailbox presentation mode.
//     auto present_mode = choose_swap_present_mode(  //
//         swapchain_support.presentModes             //
//     );
//     // Set the swap chain image sizes to be equal to the window surface
//     // sizes.
//     auto extent = choose_swap_extent(swapchain_support.capabilities);
//
//     auto image_count = swapchain_support.capabilities.maxImageCount + 1;
//     if (swapchain_support.capabilities.maxImageCount > 0 &&
//         image_count > swapchain_support.capabilities.maxImageCount)
//       image_count = swapchain_support.capabilities.maxImageCount;
//
//     auto create_info = VkSwapchainCreateInfoKHR{};
//     {
//       create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
//       create_info.surface = _surface;
//       create_info.minImageCount = image_count;
//       create_info.imageFormat = surface_format.format;
//       create_info.imageColorSpace = surface_format.colorSpace;
//       create_info.imageExtent = extent;
//       create_info.imageArrayLayers = 1;
//       create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
//     }
//
//     auto indices = find_queue_families(_physical_device);
//     auto queue_family_indices = std::array{
//         indices.graphicsFamily.value(),  //
//         indices.presentFamily.value()    //
//     };
//
//     if (indices.graphicsFamily != indices.presentFamily)
//     {
//       create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
//       create_info.queueFamilyIndexCount = 2;
//       create_info.pQueueFamilyIndices = queue_family_indices.data();
//     }
//     else
//       create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
//
//     create_info.preTransform = swapchain_support.capabilities.currentTransform;
//     create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
//     create_info.presentMode = present_mode;
//     create_info.clipped = VK_TRUE;
//     create_info.oldSwapchain = VK_NULL_HANDLE;
//
//     if (vkCreateSwapchainKHR(_device, &create_info, nullptr, &_swapchain) !=
//         VK_SUCCESS)
//       throw std::runtime_error{"Failed to create swap chain!"};
//
//     // Get the count of swap chain images.
//     vkGetSwapchainImagesKHR(_device, _swapchain, &image_count, nullptr);
//     _swapchain_images.resize(image_count);
//     // Fill the array of swap chain images.
//     vkGetSwapchainImagesKHR(_device, _swapchain, &image_count,
//                             _swapchain_images.data());
//
//     _swapchain_image_format = surface_format.format;
//     _swapchain_extent = extent;
//   }
//
//   auto query_swapchain_support(VkPhysicalDevice device) const
//       -> vk::SwapChainSupportDetails
//   {
//     auto details = vk::SwapChainSupportDetails{};
//
//     vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, _surface,
//                                               &details.capabilities);
//
//     auto format_count = std::uint32_t{};
//     vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &format_count,
//                                          nullptr);
//     if (format_count != 0)
//     {
//       details.formats.resize(format_count);
//       vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &format_count,
//                                            details.formats.data());
//     }
//
//     auto present_mode_count = std::uint32_t{};
//     vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface,
//                                               &present_mode_count, nullptr);
//     if (present_mode_count != 0)
//     {
//       details.presentModes.resize(present_mode_count);
//       vkGetPhysicalDeviceSurfacePresentModesKHR(
//           device, _surface, &present_mode_count, details.presentModes.data());
//     }
//
//     return details;
//   }
//
//   // We want a surface with the BGRA 32bit pixel format preferably.
//   auto choose_swap_surface_format(
//       const std::vector<VkSurfaceFormatKHR>& available_formats) const
//       -> VkSurfaceFormatKHR
//   {
//     for (const auto& available_format : available_formats)
//     {
//       if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
//           available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
//         return available_format;
//     }
//
//     return available_formats[0];
//   }
//
//   // Choose the simplest (not the best one) swap present mode.
//   auto choose_swap_present_mode(
//       const std::vector<VkPresentModeKHR>& available_present_modes) const
//       -> VkPresentModeKHR
//   {
//     for (const auto& available_present_mode : available_present_modes)
//       if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
//         return available_present_mode;
//
//     // This is always available in any vulkan vendor implementation.
//     // Preferred in mobile development.
//     return VK_PRESENT_MODE_FIFO_KHR;
//   }
//
//   // Choose the swap image sizes.
//   auto choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities) const
//       -> VkExtent2D
//   {
//     if (capabilities.currentExtent.width != UINT32_MAX)
//       return capabilities.currentExtent;
//
//     auto w = int{};
//     auto h = int{};
//     glfwGetFramebufferSize(_window, &w, &h);
//
//     auto actual_extent = VkExtent2D{
//         .width = std::clamp(static_cast<std::uint32_t>(w),
//                             capabilities.minImageExtent.width,
//                             capabilities.maxImageExtent.width),
//         .height = std::clamp(static_cast<std::uint32_t>(h),
//                              capabilities.minImageExtent.height,
//                              capabilities.maxImageExtent.height)  //
//     };
//
//     return actual_extent;
//   }
//
//   auto create_image_views() -> void
//   {
//     _swapchain_image_views.resize(_swapchain_images.size());
//
//     for (auto i = 0u; i < _swapchain_images.size(); ++i)
//     {
//       auto create_info = VkImageViewCreateInfo{};
//       create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
//       create_info.image = _swapchain_images[i];
//       create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
//       create_info.format = _swapchain_image_format;
//       create_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY,  //
//                                 VK_COMPONENT_SWIZZLE_IDENTITY,
//                                 VK_COMPONENT_SWIZZLE_IDENTITY,
//                                 VK_COMPONENT_SWIZZLE_IDENTITY};
//       create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
//       create_info.subresourceRange.baseMipLevel = 0;
//       create_info.subresourceRange.levelCount = 1;
//       create_info.subresourceRange.baseArrayLayer = 0;
//       create_info.subresourceRange.layerCount = 1;
//
//       if (vkCreateImageView(_device, &create_info, nullptr,
//                             &_swapchain_image_views[i]) != VK_SUCCESS)
//         throw std::runtime_error{"Failed to create image views!"};
//     }
//   }
//
//   auto create_render_pass() -> void
//   {
//     auto color_attachment = VkAttachmentDescription{};
//     {
//       color_attachment.format = _swapchain_image_format;
//       color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
//       color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
//       color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
//       color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
//       color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
//     };
//
//     auto color_attachment_ref = VkAttachmentReference{
//         .attachment = 0,                                    //
//         .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  //
//     };
//
//     auto subpass = VkSubpassDescription{};
//     {
//       subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
//       subpass.colorAttachmentCount = 1;
//       subpass.pColorAttachments = &color_attachment_ref;
//     };
//
//     auto dependency = VkSubpassDependency{};
//     {
//       dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
//       dependency.dstSubpass = 0;
//       dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
//       dependency.srcAccessMask = 0;
//       dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
//       dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
//     }
//
//
//     auto render_pass_create_info = VkRenderPassCreateInfo{};
//     {
//       render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
//       render_pass_create_info.attachmentCount = 1;
//       render_pass_create_info.pAttachments = &color_attachment;
//       render_pass_create_info.subpassCount = 1;
//       render_pass_create_info.pSubpasses = &subpass;
//       render_pass_create_info.dependencyCount = 1;
//       render_pass_create_info.pDependencies = &dependency;
//     };
//
//     if (vkCreateRenderPass(_device, &render_pass_create_info, nullptr,
//                            &_render_pass) != VK_SUCCESS)
//       throw std::runtime_error{"Failed to create render pass!"};
//   }
//
//
//   //
//   auto create_graphics_pipeline() -> void
//   {
//     const auto p = fs::path{_program_path};
//     auto vertex_shader_code = read_shader_file((p / "vert.spv").string());
//     auto fragment_shader_code = read_shader_file((p / "frag.spv").string());
//
//     auto vertex_shader_module = create_shader_module(vertex_shader_code);
//     auto fragment_shader_module = create_shader_module(fragment_shader_code);
//
//     auto vertex_shader_stage_info = VkPipelineShaderStageCreateInfo{};
//     {
//       vertex_shader_stage_info.sType =
//           VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//       vertex_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
//       vertex_shader_stage_info.module = vertex_shader_module;
//       vertex_shader_stage_info.pName = "main";
//     }
//
//     auto fragment_shader_stage_info = VkPipelineShaderStageCreateInfo{};
//     {
//       fragment_shader_stage_info.sType =
//           VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//       fragment_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
//       fragment_shader_stage_info.module = fragment_shader_module;
//       fragment_shader_stage_info.pName = "main";
//     }
//
//     auto shader_stages = std::array{
//         vertex_shader_stage_info,   //
//         fragment_shader_stage_info  //
//     };
//
//     const auto binding_description = Vertex::get_binding_description();
//     const auto attribute_description = Vertex::get_attributes_description();
//
//     auto vertex_input_info = VkPipelineVertexInputStateCreateInfo{};
//     {
//       vertex_input_info.sType =
//           VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
//       vertex_input_info.vertexBindingDescriptionCount = 1;
//       vertex_input_info.pVertexBindingDescriptions = &binding_description;
//       vertex_input_info.vertexAttributeDescriptionCount =
//           static_cast<std::uint32_t>(attribute_description.size());
//       vertex_input_info.pVertexAttributeDescriptions =
//           attribute_description.data();
//     };
//
//     auto input_assembly = VkPipelineInputAssemblyStateCreateInfo{};
//     {
//       input_assembly.sType =
//           VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
//       input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
//       input_assembly.primitiveRestartEnable = VK_FALSE;
//     };
//
//     auto viewport = VkViewport{
//         .x = 0.f,
//         .y = 0.f,
//         .width = static_cast<float>(_swapchain_extent.width),
//         .height = static_cast<float>(_swapchain_extent.height),
//         .minDepth = 0.f,
//         .maxDepth = 1.f  //
//     };
//
//     auto scissor = VkRect2D{
//         .offset = {0, 0},
//         .extent = _swapchain_extent  //
//     };
//
//     auto viewport_state = VkPipelineViewportStateCreateInfo{};
//     {
//       viewport_state.sType =
//           VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
//       viewport_state.pNext = nullptr;
//       viewport_state.viewportCount = 1;
//       viewport_state.pViewports = &viewport;
//       viewport_state.scissorCount = 1;
//       viewport_state.pScissors = &scissor;
//     };
//
//     auto rasterizer = VkPipelineRasterizationStateCreateInfo{};
//     {
//       rasterizer.sType =
//           VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
//       rasterizer.depthClampEnable = VK_FALSE;
//       rasterizer.rasterizerDiscardEnable = VK_FALSE;  //
//       rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
//       rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
//       rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
//       rasterizer.depthBiasEnable = VK_FALSE;
//       rasterizer.depthBiasConstantFactor = 0.f;
//       rasterizer.depthBiasSlopeFactor = 0.f;
//       rasterizer.lineWidth = 1.f;
//     }
//
//     auto multisampling = VkPipelineMultisampleStateCreateInfo{};
//     {
//       multisampling.sType =
//           VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
//       multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
//       multisampling.sampleShadingEnable = VK_FALSE;
//       multisampling.minSampleShading = 1.f;
//       multisampling.pSampleMask = nullptr;
//       multisampling.alphaToCoverageEnable = VK_FALSE;
//       multisampling.alphaToOneEnable = VK_FALSE;
//     }
//
//     auto color_blend_attachment = VkPipelineColorBlendAttachmentState{};
//     {
//       color_blend_attachment.blendEnable = VK_FALSE;
//       color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
//       color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
//       color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
//       color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
//       color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;  //
//       color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |  //
//                                               VK_COLOR_COMPONENT_G_BIT |  //
//                                               VK_COLOR_COMPONENT_B_BIT |  //
//                                               VK_COLOR_COMPONENT_A_BIT;   //
//     }
//
//     auto color_blending = VkPipelineColorBlendStateCreateInfo{};
//     {
//       color_blending.sType =
//           VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
//       color_blending.logicOpEnable = VK_FALSE;
//       color_blending.logicOp = VK_LOGIC_OP_COPY;
//       color_blending.attachmentCount = 1;
//       color_blending.pAttachments = &color_blend_attachment;
//       for (auto i = 0; i < 4; ++i)
//         color_blending.blendConstants[i] = 0.f;
//     };
//
//     auto pipeline_layout_info = VkPipelineLayoutCreateInfo{};
//     {
//       pipeline_layout_info.sType =
//           VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
//       pipeline_layout_info.setLayoutCount = 0;
//       pipeline_layout_info.pushConstantRangeCount = 0;
//     };
//
//     if (vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr,
//                                &_pipeline_layout) != VK_SUCCESS)
//       throw std::runtime_error{"Failed to create pipeline layout!"};
//
//     auto pipeline_info = VkGraphicsPipelineCreateInfo{};
//     {
//       pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
//       pipeline_info.stageCount = shader_stages.size();
//       pipeline_info.pStages = shader_stages.data();
//       pipeline_info.pVertexInputState = &vertex_input_info;
//       pipeline_info.pInputAssemblyState = &input_assembly;
//       pipeline_info.pViewportState = &viewport_state;
//       pipeline_info.pRasterizationState = &rasterizer;
//       pipeline_info.pMultisampleState = &multisampling;
//       pipeline_info.pColorBlendState = &color_blending;
//       pipeline_info.layout = _pipeline_layout;
//       pipeline_info.renderPass = _render_pass;
//       pipeline_info.subpass = 0;
//       pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
//       pipeline_info.basePipelineIndex = -1;
//     };
//
//     if (vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info,
//                                   nullptr, &_graphics_pipeline) != VK_SUCCESS)
//       throw std::runtime_error{"Failed to create graphics pipeline!"};
//
//     vkDestroyShaderModule(_device, vertex_shader_module, nullptr);
//     vkDestroyShaderModule(_device, fragment_shader_module, nullptr);
//   }
//
//   static auto read_shader_file(const std::string& filename) -> std::vector<char>
//   {
//     std::ifstream file(filename, std::ios::ate | std::ios::binary);
//     if (!file)
//       throw std::runtime_error{"Failed to open file!"};
//
//     const auto file_size = static_cast<std::size_t>(file.tellg());
//     auto buffer = std::vector<char>(file_size);
//
//     file.seekg(0);
//     file.read(buffer.data(), file_size);
//     file.close();
//
//     return buffer;
//   }
//
//   auto create_shader_module(const std::vector<char>& buffer) -> VkShaderModule
//   {
//     auto create_info = VkShaderModuleCreateInfo{};
//     {
//       create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
//       create_info.codeSize = static_cast<std::uint32_t>(buffer.size());
//       create_info.pCode = reinterpret_cast<const std::uint32_t*>(buffer.data());
//     }
//
//     auto shader_module = VkShaderModule{};
//     if (vkCreateShaderModule(_device, &create_info, nullptr, &shader_module) !=
//         VK_SUCCESS)
//       throw std::runtime_error{"Failed to create shader module!"};
//
//     return shader_module;
//   }
//
//
//   //
//   auto create_framebuffers() -> void
//   {
//     _swapchain_framebuffers.resize(_swapchain_image_views.size());
//
//     for (auto i = 0u; i < _swapchain_image_views.size(); ++i)
//     {
//       auto framebuffer_info = VkFramebufferCreateInfo{};
//       {
//         framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
//         framebuffer_info.renderPass = _render_pass;
//         framebuffer_info.attachmentCount = 1;
//         framebuffer_info.pAttachments = &_swapchain_image_views[i];
//         framebuffer_info.width = _swapchain_extent.width;
//         framebuffer_info.height = _swapchain_extent.height;
//         framebuffer_info.layers = 1;
//       }
//
//       if (vkCreateFramebuffer(_device, &framebuffer_info, nullptr,
//                               &_swapchain_framebuffers[i]) != VK_SUCCESS)
//         throw std::runtime_error{"Failed to create framebuffer!"};
//     }
//   }
//
//
//   // Command pool for graphics family queue.
//   auto create_command_pool() -> void
//   {
//     const auto queue_family_indices = find_queue_families(_physical_device);
//
//     auto pool_info = VkCommandPoolCreateInfo{};
//     {
//       pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
//       pool_info.queueFamilyIndex = queue_family_indices.graphicsFamily.value();
//       pool_info.flags = 0;
//     }
//
//     if (vkCreateCommandPool(_device, &pool_info, nullptr, &_command_pool) !=
//         VK_SUCCESS)
//       throw std::runtime_error{"Failed to create command pool!"};
//   }
//
//   auto create_command_buffers() -> void
//   {
//     _command_buffers.resize(_swapchain_framebuffers.size());
//
//     auto alloc_info = VkCommandBufferAllocateInfo{};
//     {
//       alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
//       alloc_info.commandPool = _command_pool;
//       alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
//       alloc_info.commandBufferCount =
//           static_cast<std::uint32_t>(_command_buffers.size());
//     }
//
//     if (vkAllocateCommandBuffers(_device, &alloc_info,
//                                  _command_buffers.data()) != VK_SUCCESS)
//       throw std::runtime_error{"Failed to allocate command buffers!"};
//
//     for (auto i = 0u; i < _command_buffers.size(); ++i)
//     {
//       auto begin_info = VkCommandBufferBeginInfo{};
//       begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
//       begin_info.flags = 0;
//       begin_info.pInheritanceInfo = nullptr;
//
//       if (vkBeginCommandBuffer(_command_buffers[i], &begin_info) != VK_SUCCESS)
//         throw std::runtime_error{"Failed to begin recording command buffer!"};
//
//       auto render_pass_begin_info = VkRenderPassBeginInfo{};
//       {
//         render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
//         render_pass_begin_info.renderPass = _render_pass;
//         render_pass_begin_info.framebuffer = _swapchain_framebuffers[i];
//         render_pass_begin_info.renderArea.offset = {0, 0};
//         render_pass_begin_info.renderArea.extent = _swapchain_extent;
//
//         auto clear_color = VkClearValue{{{0.f, 0.f, 0.f, 1.f}}};
//         render_pass_begin_info.clearValueCount = 1;
//         render_pass_begin_info.pClearValues = &clear_color;
//       }
//
//       vkCmdBeginRenderPass(_command_buffers[i], &render_pass_begin_info,
//                            VK_SUBPASS_CONTENTS_INLINE);
//       {
//         vkCmdBindPipeline(_command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
//                           _graphics_pipeline);
//         vkCmdDraw(_command_buffers[i], 3, 1, 0, 0);
//       }
//       vkCmdEndRenderPass(_command_buffers[i]);
//
//       if (vkEndCommandBuffer(_command_buffers[i]) != VK_SUCCESS)
//         throw std::runtime_error{"Failed to record command buffer!"};
//     }
//   }
//
//
//   auto create_sync_objects() -> void
//   {
//     _image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
//     _render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
//     _in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
//     _images_in_flight.resize(_swapchain_images.size(), VK_NULL_HANDLE);
//
//     auto semaphore_info = VkSemaphoreCreateInfo{};
//     semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
//
//     auto fence_info = VkFenceCreateInfo{};
//     fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
//     fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
//
//     for (auto i = 0u; i < MAX_FRAMES_IN_FLIGHT; ++i)
//     {
//       if (vkCreateSemaphore(_device, &semaphore_info, nullptr,
//                             &_image_available_semaphores[i]) != VK_SUCCESS ||
//           vkCreateSemaphore(_device, &semaphore_info, nullptr,
//                             &_render_finished_semaphores[i]) != VK_SUCCESS ||
//           vkCreateFence(_device, &fence_info, nullptr, &_in_flight_fences[i]) !=
//               VK_SUCCESS)
//       {
//         throw std::runtime_error{
//             "Failed to create synchronization objects for a frame!"};
//       }
//     }
//   }
//
//
//   //
//   auto draw_frame() -> void
//   {
//     // The GPU is blocking the CPU at this point of the road:
//     // - The CPU is stopped at a fence
//     // - The first fence is closed
//     vkWaitForFences(_device, 1, &_in_flight_fences[current_frame], VK_TRUE,
//                     UINT64_MAX);
//     // - The first fence has opened just now.
//
//     // The CPU pursues its journey:
//     // - it acquires the next image for rendering.
//     auto image_index = std::uint32_t{};
//     auto result =
//         vkAcquireNextImageKHR(_device, _swapchain, UINT64_MAX,
//                               _image_available_semaphores[current_frame],
//                               VK_NULL_HANDLE, &image_index);
//     if (result == VK_ERROR_OUT_OF_DATE_KHR)
//     {
//       // For example when the window is resized, we need to recreate the swap
//       // chain, the images of the swap chains must reinitialized with the new
//       // window extent.
//       recreate_swapchain();
//       return;
//     }
//     else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
//       throw std::runtime_error("failed to acquire swap chain image!");
//
//     // The CPU encounters another fence controlled by the GPU:
//     // - The CPU is waiting from the GPU until the image becomes available for
//     //   rendering.
//     // - The second fence displays a red light
//     // - When rendering for the first time, the second fence is already opened,
//     //   so the CPU can pursue its journey.
//     if (_images_in_flight[image_index] != VK_NULL_HANDLE)
//       vkWaitForFences(_device, 1, &_images_in_flight[current_frame], VK_TRUE,
//                       UINT64_MAX);
//     _images_in_flight[image_index] = _in_flight_fences[current_frame];
//     // - The second fence has opened just now.
//
//     // The CPU pursues its journey:
//     //
//     // - The CPU will now detail what to do at the drawing stage.
//     auto submit_info = VkSubmitInfo{};
//     submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
//
//     VkPipelineStageFlags wait_stages =
//         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
//     // - The CPU tells the GPU to ensure that it starts drawing only when the
//     // image becomes available.
//     submit_info.waitSemaphoreCount = 1;
//     submit_info.pWaitSemaphores = &_image_available_semaphores[current_frame];
//     submit_info.pWaitDstStageMask = &wait_stages;
//
//     // - The CPU tells the GPU to ensure that it notifies when the drawing is
//     //   finished and thus ready to present onto the screen.
//     submit_info.signalSemaphoreCount = 1;
//     submit_info.pSignalSemaphores = &_render_finished_semaphores[current_frame];
//
//     submit_info.commandBufferCount = 1;
//     submit_info.pCommandBuffers = &_command_buffers[image_index];
//
//     // - The CPU now tells the GPU to make the first fence close the road.
//     vkResetFences(_device, 1, &_in_flight_fences[current_frame]);
//
//     // - The CPU submits a drawing command to the GPU (on the graphics queue).
//     if (vkQueueSubmit(_graphics_queue, 1, &submit_info,
//                       _in_flight_fences[current_frame]) != VK_SUCCESS)
//       throw std::runtime_error{"Failed to submit draw command buffer!"};
//
//     // - The CPU details the screen presentation command.
//     auto present_info = VkPresentInfoKHR{};
//     {
//       present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
//
//       // - The CPU tells the GPU to trigger the frame display only when the
//       //   rendering is finished.
//       //
//       //   This completely specifies the dependency between the rendering
//       //   command and the screen display command.
//       present_info.waitSemaphoreCount = 1;
//       present_info.pWaitSemaphores =
//           &_render_finished_semaphores[current_frame];
//
//       present_info.swapchainCount = 1;
//       present_info.pSwapchains = &_swapchain;
//
//       present_info.pImageIndices = &image_index;
//     }
//
//     // - The CPU submits another command to the GPU (on the present queue).
//     vkQueuePresentKHR(_present_queue, &present_info);
//
//     // Move to the next framebuffer.
//     current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
//   }
//
//
//   //
//   auto recreate_swapchain() -> void
//   {
//     vkDeviceWaitIdle(_device);
//
//     cleanup_swapchain();
//
//     create_swapchain();
//     create_image_views();
//     create_render_pass();
//     create_graphics_pipeline();
//     create_framebuffers();
//     create_command_buffers();
//   }
//
//   auto cleanup_swapchain() -> void
//   {
//     for (auto& framebuffer : _swapchain_framebuffers)
//       vkDestroyFramebuffer(_device, framebuffer, nullptr);
//
//     vkFreeCommandBuffers(_device, _command_pool,
//                          static_cast<std::uint32_t>(_command_buffers.size()),
//                          _command_buffers.data());
//
//     vkDestroyPipeline(_device, _graphics_pipeline, nullptr);
//     vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
//     vkDestroyRenderPass(_device, _render_pass, nullptr);
//
//     for (auto& image_view : _swapchain_image_views)
//       vkDestroyImageView(_device, image_view, nullptr);
//
//     vkDestroySwapchainKHR(_device, _swapchain, nullptr);
//   }
//
//   std::string _program_path;
//   GLFWwindow* _window = nullptr;
//
//   VkInstance _instance;
//   VkDebugUtilsMessengerEXT _debug_messenger;
//   VkSurfaceKHR _surface;
//
//   VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
//   VkDevice _device;
//
//   VkQueue _graphics_queue;
//   VkQueue _present_queue;
//
//   VkSwapchainKHR _swapchain;
//   std::vector<VkImage> _swapchain_images;
//   VkFormat _swapchain_image_format;
//   VkExtent2D _swapchain_extent;
//   std::vector<VkImageView> _swapchain_image_views;
//   std::vector<VkFramebuffer> _swapchain_framebuffers;
//
//   VkRenderPass _render_pass;
//   VkPipelineLayout _pipeline_layout;
//   VkPipeline _graphics_pipeline;
//
//   VkCommandPool _command_pool;
//   std::vector<VkCommandBuffer> _command_buffers;
//
//   //! @brief "Traffic lights".
//   //! @{
//   std::vector<VkSemaphore> _image_available_semaphores;
//   std::vector<VkSemaphore> _render_finished_semaphores;
//   std::vector<VkFence> _in_flight_fences;
//   std::vector<VkFence> _images_in_flight;
//   //! @}
//
//   std::int32_t current_frame = 0;
// };


int main(int, char**)
{
//   const auto program_path =
//       fs::canonical(fs::system_complete(argv[0]).parent_path());
//   HelloTriangleApplication app{program_path.string()};
//
//   try
//   {
//     app.run();
//   }
//   catch (const std::exception& e)
//   {
//     std::cerr << e.what() << std::endl;
//     return EXIT_FAILURE;
//   }

  return 0;
}
