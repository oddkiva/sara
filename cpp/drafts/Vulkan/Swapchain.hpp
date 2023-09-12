#pragma once

#define GLFW_INCLUDE_VULKAN

#include <drafts/Vulkan/VulkanGLFWInterop.hpp>

#include <drafts/Vulkan/Device.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <GLFW/glfw3.h>


namespace DO::Kalpana::Vulkan {

  struct SwapChainSupportDetails
  {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
  };

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

  inline auto query_swapchain_support(const VkPhysicalDevice physical_device,
                                      const VkSurfaceKHR surface)
      -> SwapChainSupportDetails
  {
    auto details = SwapChainSupportDetails{};
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

  struct Swapchain
  {
    Swapchain() = default;

    Swapchain(const Shakti::Vulkan::PhysicalDevice& physical_device,
              const Shakti::Vulkan::Device& device, const Surface& surface,
              GLFWwindow* window)
    {
      SARA_DEBUG << "[VK] Check the physical device support swapchains...\n";
      auto placeholder = query_swapchain_support;
      const auto swapchain_support = placeholder(  //
          physical_device,                         //
          surface                                  //
      );

      // Find a valid pixel format for the swap chain images and if possible
      // choose RGBA 32 bits.
      SARA_DEBUG << "[VK] Setting the swap surface format (RGBA 32-bit)...\n";
      const auto surface_format =
          choose_swap_surface_format(swapchain_support.formats);
      // FIFO/Mailbox presentation mode.
      SARA_DEBUG << "[VK] Setting the swap present mode...\n";
      auto present_mode = choose_swap_present_mode(  //
          swapchain_support.present_modes            //
      );
      // Set the swap chain image sizes to be equal to the window surface
      // sizes.
      SARA_DEBUG << "[VK] Setting the swap extent...\n";
      swapchain_extent =
          choose_swap_extent(window, swapchain_support.capabilities);

      SARA_DEBUG << "[VK] Setting the swap image count...\n";
      auto image_count = swapchain_support.capabilities.maxImageCount + 1;
      if (swapchain_support.capabilities.maxImageCount > 0 &&
          image_count > swapchain_support.capabilities.maxImageCount)
        image_count = swapchain_support.capabilities.maxImageCount;

      SARA_DEBUG << "[VK] Initializing the swapchain...\n";
      auto create_info = VkSwapchainCreateInfoKHR{};
      {
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = surface;
        create_info.minImageCount = image_count;
        create_info.imageFormat = surface_format.format;
        create_info.imageColorSpace = surface_format.colorSpace;
        create_info.imageExtent = swapchain_extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      }

      const VkPhysicalDevice physical_device_handle = physical_device;
      const auto graphics_queue_family_index =
          find_graphics_queue_family_indices(physical_device_handle).front();
      const auto present_queue_family_index =
          find_present_queue_family_indices(physical_device_handle, surface)
              .front();
      const auto queue_family_indices = std::array{
          graphics_queue_family_index,  //
          present_queue_family_index    //
      };

      if (graphics_queue_family_index != present_queue_family_index)
      {
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = queue_family_indices.data();
      }
      else
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

      create_info.preTransform =
          swapchain_support.capabilities.currentTransform;
      create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
      create_info.presentMode = present_mode;
      create_info.clipped = VK_TRUE;
      create_info.oldSwapchain = VK_NULL_HANDLE;

      const status =
          vkCreateSwapchainKHR(device.handle, &create_info, nullptr, &_handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{fmt::format(
            "Error: failed to create Vulkan swapchain! Error code: {}",
            static_cast<int>(status))};

      SARA_DEBUG << "[VK] Initializing the swapchain images...\n";
      swapchain_images = list_swapchain_images(device.handle, handle);

      SARA_DEBUG << "[VK] Initializing the swapchain image format...\n";
      swapchain_image_format = surface_format.format;
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
    auto choose_swap_extent(GLFWwindow* window,
                            const VkSurfaceCapabilitiesKHR& capabilities) const
        -> VkExtent2D
    {
      if (capabilities.currentExtent.width != UINT32_MAX)
        return capabilities.currentExtent;

      auto w = int{};
      auto h = int{};
      glfwGetFramebufferSize(window, &w, &h);

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

    VkSwapchainKHR handle;
    std::vector<VkImage> swapchain_images;
    VkFormat swapchain_image_format;
    VkExtent2D swapchain_extent;

    std::vector<VkImageView> _swapchain_image_views;
    std::vector<VkFramebuffer> _swapchain_framebuffers;
  };

}  // namespace DO::Kalpana::Vulkan
