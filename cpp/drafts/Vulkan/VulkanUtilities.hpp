#pragma once

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include <fmt/format.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>
#include <vulkan/vulkan_core.h>


namespace vk {

  struct QueueFamilyIndices
  {
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> present_family;

    // Check if the GPU driver supports Vulkan. By that we mean we should be
    // able to:
    // - display stuff on the screen (present queue)
    // - perform typical computer graphics operations (graphics queue).
    auto is_complete() const -> bool
    {
      return graphics_family.has_value() && present_family.has_value();
    }
  };

  struct SwapChainSupportDetails
  {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
  };


  //! Debug messenger hooks to the Vulkan instance
  //! @{
  inline auto
  get_required_extensions_from_glfw(const bool enable_validation_layers)
      -> std::vector<const char*>
  {
    auto glfw_extension_count = std::uint32_t{};
    const char** glfw_extensions = nullptr;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    std::vector<const char*> extensions(glfw_extensions,
                                        glfw_extensions + glfw_extension_count);

    if (enable_validation_layers)
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
  }

  inline auto create_debug_utils_messenger_ext(
      const VkInstance instance,
      const VkDebugUtilsMessengerCreateInfoEXT* create_info,
      const VkAllocationCallbacks* allocator,
      VkDebugUtilsMessengerEXT* debug_messenger) -> VkResult
  {
    // Check if the vendor implemented this function.
    const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (func == nullptr)
      return VK_ERROR_EXTENSION_NOT_PRESENT;

    // Then call if if exists.
    return func(instance, create_info, allocator, debug_messenger);
  }

  inline auto destroy_debug_utils_messenger_ext(
      const VkInstance instance, const VkDebugUtilsMessengerEXT debug_messenger,
      const VkAllocationCallbacks* allocator) -> void
  {
    const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func == nullptr)
      return;

    return func(instance, debug_messenger, allocator);
  }

  template <typename STLArrayOfStrings>
  auto check_validation_layer_support(
      const STLArrayOfStrings& requested_validation_layers) -> bool
  {
    auto layer_count = std::uint32_t{};
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    if (layer_count == 0 && !requested_validation_layers.empty())
      return false;

    auto available_layers = std::vector<VkLayerProperties>(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    for (const char* layer_name : requested_validation_layers)
    {
      const auto available_layer_it = std::find_if(
          available_layers.begin(), available_layers.end(),
          [&layer_name](const auto& layer_properties) {
            return strcmp(layer_name, layer_properties.layerName) == 0;
          });
      const auto layer_found = available_layer_it != available_layers.end();

      std::cout << fmt::format("  [VK][Validation] {}: {}\n",  //
                               layer_name, layer_found ? "FOUND" : "NOT FOUND");

      if (!layer_found)
        return false;
    }

    return true;
  }

  inline VKAPI_ATTR auto VKAPI_CALL
  debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                 VkDebugUtilsMessageTypeFlagsEXT message_type,
                 const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                 void* /* user_data */) -> VkBool32
  {

    std::cerr << fmt::format("  [VK-Debug][SEV {:03d}][TYPE {:02d}] {}\n",  //
                             static_cast<std::uint32_t>(message_severity),
                             static_cast<std::uint32_t>(message_type),
                             callback_data->pMessage);
    return VK_FALSE;
  }

  inline auto init_debug_messenger_create_info(
      VkDebugUtilsMessengerCreateInfoEXT& create_info) -> void
  {
    create_info = VkDebugUtilsMessengerCreateInfoEXT{};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debug_callback;
  }
  //! @}


  inline auto list_physical_devices(const VkInstance instance)
      -> std::vector<VkPhysicalDevice>
  {
    SARA_DEBUG << "  [VK] Counting the number of physical devices...\n";
    auto device_count = std::uint32_t{};
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    SARA_DEBUG << fmt::format("  [VK] Physical device count: {}\n",
                              device_count);

    SARA_DEBUG << "  [VK] Populating the list of physical devices...\n";
    if (device_count == 0)
      return {};

    auto devices = std::vector<VkPhysicalDevice>(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    return devices;
  }

  inline auto list_device_extensions(const VkPhysicalDevice device)
      -> std::vector<VkExtensionProperties>
  {
    auto extension_count = std::uint32_t{};
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                         nullptr);

    if (extension_count == 0)
      return {};

    auto extensions = std::vector<VkExtensionProperties>(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                         extensions.data());
    return extensions;
  }

  inline auto query_swapchain_support(const VkPhysicalDevice physical_device,
                                      const VkSurfaceKHR surface)
      -> vk::SwapChainSupportDetails
  {
    auto details = vk::SwapChainSupportDetails{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,
                                              &details.capabilities);

    auto format_count = std::uint32_t{};
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface,
                                         &format_count, nullptr);
    if (format_count != 0)
    {
      details.formats.resize(format_count);
      vkGetPhysicalDeviceSurfaceFormatsKHR(
          physical_device, surface, &format_count, details.formats.data());
    }

    auto present_mode_count = std::uint32_t{};
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                              &present_mode_count, nullptr);
    if (present_mode_count != 0)
    {
      details.present_modes.resize(present_mode_count);
      vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                                &present_mode_count,
                                                details.present_modes.data());
    }

    return details;
  }

  inline auto list_swapchain_images(const VkDevice device,
                                    VkSwapchainKHR swapchain)
      -> std::vector<VkImage>
  {
    // Get the count of swapchain images.
    auto swapchain_image_count = std::uint32_t{};
    vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, nullptr);

    // Populate the array of swapchain images.
    auto swapchain_images = std::vector<VkImage>(swapchain_image_count);
    vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count,
                            swapchain_images.data());

    return swapchain_images;
  }


}  // namespace vk
