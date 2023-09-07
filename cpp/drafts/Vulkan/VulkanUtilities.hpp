#pragma once

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

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
      auto layer_found = false;

      for (const auto& layer_properties : available_layers)
      {
        SARA_CHECK(layer_name);
        SARA_CHECK(layer_properties.layerName);
        if (strcmp(layer_name, layer_properties.layerName) == 0)
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

  inline VKAPI_ATTR auto VKAPI_CALL
  debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                 VkDebugUtilsMessageTypeFlagsEXT message_type,
                 const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                 void* /* user_data */) -> VkBool32
  {

    std::cerr << "[DEBUG]"
              << "[" << message_severity << "]"
              << "[" << message_type << "]"  //
              << "Validation layer: " << callback_data->pMessage << std::endl;
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

}  // namespace vk
