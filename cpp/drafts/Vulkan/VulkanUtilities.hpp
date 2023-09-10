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


namespace DO::Kalpana::EasyVulkan {

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


}  // namespace DO::Kalpana::EasyVulkan
