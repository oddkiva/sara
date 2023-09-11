// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <drafts/Vulkan/Device.hpp>
#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>

#include <drafts/Vulkan/VulkanGLFWInterop.hpp>


namespace DO::Kalpana::Vulkan {

  class VulkanGraphicsBackend
  {
  public:
#if defined(__APPLE__)
    static constexpr auto compile_for_apple = true;
#else
    static constexpr auto compiling_for_apple = false;
#endif

    static constexpr auto default_width = 800;
    static constexpr auto default_height = 600;

  public:
    VulkanGraphicsBackend(const std::string& app_name, const bool debug_vulkan)
    {
      init_instance(app_name, debug_vulkan);
    }

    auto init_instance(const std::string& app_name, const bool debug_vulkan)
        -> void
    {
      // Vulkan instance.
      _instance_extensions = list_required_vulkan_extensions_from_glfw();
      if constexpr (compile_for_apple)
      {
        _instance_extensions.emplace_back(
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        _instance_extensions.emplace_back(
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
      }
      if (debug_vulkan)
        _instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

      if (debug_vulkan)
        _validation_layers = std::vector{
            "VK_LAYER_KHRONOS_validation"  //
        };

      _instance = Shakti::Vulkan::InstanceCreator{}
                      .application_name(app_name.c_str())
                      .engine_name("No Engine")
                      .enable_instance_extensions(_instance_extensions)
                      .enable_validation_layers(_validation_layers)
                      .create();
    }

    auto init_surface(GLFWwindow* window) -> void
    {
      _surface.init(_instance, window);
    }

    auto init_physical_device() -> void
    {
      namespace svk = Shakti::Vulkan;

      // List all Vulkan physical devices.
      const auto physical_devices =
          svk::PhysicalDevice::list_physical_devices(_instance);

      // Find a suitable physical (GPU) device that can be used for 3D graphics
      // application.
      const auto di = std::find_if(
          physical_devices.begin(), physical_devices.end(),
          [this](const auto& d) {
            return d.supports_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME) &&
                   !find_graphics_queue_family_indices(d).empty() &&
                   !find_present_queue_family_indices(d, _surface).empty();
          });
      if (di == physical_devices.end())
      {
        std::cerr
            << "[VK] Error: could not find suitable Vulkan physical device!";
        return;
      }

      // There must be a suitable GPU device...
      _physical_device = *di;
    }

    auto init_device_and_queues() -> void
    {
      namespace svk = Shakti::Vulkan;

      // According to:
      // https://stackoverflow.com/questions/61434615/in-vulkan-is-it-beneficial-for-the-graphics-queue-family-to-be-separate-from-th
      //
      // Using distinct queue families, namely one for the graphics operations
      // and another for the present operations, does not result in better
      // performance.
      //
      // This is because the hardware does not expose present-only queue
      // families...
      _graphics_queue_family_index =
          find_graphics_queue_family_indices(_physical_device).front();
      _present_queue_family_index =
          find_present_queue_family_indices(_physical_device, _surface).front();

      // Create a logical device.
      auto device_extensions = std::vector{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
      if constexpr (compile_for_apple)
        device_extensions.emplace_back("VK_KHR_portability_subset");
      _device = svk::DeviceCreator{_physical_device}
                    .enable_device_extensions(device_extensions)
                    .enable_queue_families({graphics_queue_family_index,
                                            present_queue_family_index})
                    .enable_device_features({})
                    .enable_validation_layers(_validation_layers)
                    .create();

      SARA_DEBUG
          << "[VK] - Fetching the graphics queue from the logical device...\n";
      vkGetDeviceQueue(_device, graphics_queue_family_index, 0,
                       &_graphics_queue);
      SARA_DEBUG
          << "[VK] - Fetching the present queue from the logical device...\n";
      vkGetDeviceQueue(_device, present_queue_family_index, 0, &_present_queue);
    }

    auto cleanup() -> void
    {
      // Destroy in this order.
      _surface.destroy(_instance);
    }

  private:
    // The Vulkan instance.
    std::vector<const char*> _instance_extensions;
    std::vector<const char*> _validation_layers;
    Shakti::Vulkan::Instance _instance;

    // The Vulkan rendering surface.
    Surface _surface;

    // The Vulkan-compatible GPU device.
    Shakti::Vulkan::PhysicalDevice _physical_device;

    // The Vulkan logical device to which the physical device is bound.
    Shakti::Vulkan::Device _device;

    // The Vulkan capabilities that the logical device needs to have:
    // - Graphics rendering operations
    // - Display operations
    //
    // N.B.: no need to destroy these objects.
    VkQueue _graphics_queue;
    VkQueue _present_queue;
  };


}  // namespace DO::Kalpana::Vulkan
