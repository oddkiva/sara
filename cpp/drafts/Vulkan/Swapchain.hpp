#pragma once

#define GLFW_INCLUDE_VULKAN

#include <drafts/Vulkan/Device.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>
#include <drafts/Vulkan/Surface.hpp>

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
    Swapchain(const Shakti::Vulkan::PhysicalDevice& physical_device,
              const Shakti::Vulkan::Device& device,  //
              const Surface& surface,                //
              GLFWwindow* window)
      : device{device}
    {
      init_swapchain(physical_device, surface, window);
      init_image_views();
    }

    ~Swapchain()
    {
      if (handle == nullptr)
        return;

      SARA_DEBUG << "[VK] Destroying swapchain image views...\n";
      for (const auto image_view : image_views)
        vkDestroyImageView(device.handle, image_view, nullptr);

      SARA_DEBUG << "[VK] Destroying swapchain...\n";
      vkDestroySwapchainKHR(device.handle, handle, nullptr);
    }

  public: /* initialization methods */
    auto init_swapchain(const Shakti::Vulkan::PhysicalDevice& physical_device,
                        const Surface& surface, GLFWwindow* window) -> void
    {
      SARA_DEBUG
          << "[VK] Check the physical device supports the swapchain...\n";
      const auto swapchain_support = query_swapchain_support(  //
          physical_device,                                     //
          surface                                              //
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
      extent = choose_swap_extent(window, swapchain_support.capabilities);

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
        create_info.imageExtent = extent;
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

      const auto status =
          vkCreateSwapchainKHR(device.handle, &create_info, nullptr, &handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{fmt::format(
            "Error: failed to create Vulkan swapchain! Error code: {}",
            static_cast<int>(status))};

      SARA_DEBUG << "[VK] Initializing the swapchain images...\n";
      images = list_swapchain_images(device.handle, handle);

      SARA_DEBUG << "[VK] Initializing the swapchain image format...\n";
      image_format = surface_format.format;
    }

    auto init_image_views() -> void
    {
      SARA_DEBUG << "[VK] Create swapchain image views...\n";
      image_views.resize(images.size());

      for (auto i = 0u; i < images.size(); ++i)
      {
        auto create_info = VkImageViewCreateInfo{};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        create_info.image = images[i];
        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format = image_format;
        create_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY,  //
                                  VK_COMPONENT_SWIZZLE_IDENTITY,
                                  VK_COMPONENT_SWIZZLE_IDENTITY,
                                  VK_COMPONENT_SWIZZLE_IDENTITY};
        create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        create_info.subresourceRange.baseMipLevel = 0;
        create_info.subresourceRange.levelCount = 1;
        create_info.subresourceRange.baseArrayLayer = 0;
        create_info.subresourceRange.layerCount = 1;

        const auto status = vkCreateImageView(device.handle, &create_info,
                                              nullptr, &image_views[i]);
        if (status != VK_SUCCESS)
          throw std::runtime_error{
              fmt::format("[VK] Failed to create image views! Error code {}",
                          static_cast<int>(status))};
      }
    }

  public: /* opiniated configuration methods */
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

  public: /* data members */
    const Shakti::Vulkan::Device& device;

    VkSwapchainKHR handle = nullptr;
    std::vector<VkImage> images;
    VkFormat image_format;
    VkExtent2D extent;

    std::vector<VkImageView> image_views;
    std::vector<VkFramebuffer> framebuffers;
  };

}  // namespace DO::Kalpana::Vulkan
