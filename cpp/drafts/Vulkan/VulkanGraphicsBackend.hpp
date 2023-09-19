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

#include <drafts/Vulkan/CommandBuffer.hpp>
#include <drafts/Vulkan/CommandPool.hpp>
#include <drafts/Vulkan/Device.hpp>
#include <drafts/Vulkan/Fence.hpp>
#include <drafts/Vulkan/GraphicsPipeline.hpp>
#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>
#include <drafts/Vulkan/Queue.hpp>
#include <drafts/Vulkan/RenderPass.hpp>
#include <drafts/Vulkan/Semaphore.hpp>
#include <drafts/Vulkan/Surface.hpp>
#include <drafts/Vulkan/Swapchain.hpp>


namespace DO::Kalpana::Vulkan {

  class VulkanGraphicsBackend
  {
  public:
#if defined(__APPLE__)
    static constexpr auto compile_for_apple = true;
#else
    static constexpr auto compile_for_apple = false;
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
      _instance_extensions =
          Surface::list_required_instance_extensions_from_glfw();
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

      _instance = Shakti::Vulkan::Instance::Builder{}
                      .application_name(app_name.c_str())
                      .engine_name("No Engine")
                      .enable_instance_extensions(_instance_extensions)
                      .enable_validation_layers(_validation_layers)
                      .create();
    }

    auto init_surface(GLFWwindow* window) -> void
    {
      _surface = Surface{_instance, window};
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
      const auto graphics_queue_family_index =
          find_graphics_queue_family_indices(_physical_device).front();
      const auto present_queue_family_index =
          find_present_queue_family_indices(_physical_device, _surface).front();

      // Create a logical device.
      auto device_extensions = std::vector{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
      if constexpr (compile_for_apple)
        device_extensions.emplace_back("VK_KHR_portability_subset");
      _device = svk::Device::Builder{_physical_device}
                    .enable_device_extensions(device_extensions)
                    .enable_queue_families({graphics_queue_family_index,
                                            present_queue_family_index})
                    .enable_device_features({})
                    .enable_validation_layers(_validation_layers)
                    .create();

      SARA_DEBUG
          << "[VK] - Fetching the graphics queue from the logical device...\n";
      _graphics_queue = svk::Queue{_device, graphics_queue_family_index};
      SARA_DEBUG
          << "[VK] - Fetching the present queue from the logical device...\n";
      _present_queue = svk::Queue{_device, present_queue_family_index};
    }

    auto init_swapchain(GLFWwindow* window) -> void
    {
      _swapchain = Swapchain{_physical_device, _device, _surface, window};
    }

    auto init_render_pass() -> void
    {
      _render_pass.create_basic_render_pass(_device, _swapchain.image_format);
    }

    auto init_graphics_pipeline() -> void
    {
#if defined(__APPLE__)
      static const auto vs_path =
          "/Users/oddkiva/GitLab/oddkiva/sara-build-Debug/vert.spv";
      static const auto fs_path =
          "/Users/oddkiva/GitLab/oddkiva/sara-build-Debug/frag.spv";
#elif defined(_WIN32)
      static const auto vs_path =
          "C:/Users/David/Desktop/GitLab/sara-build-vs2022-static/vert.spv";
      static const auto fs_path =
          "C:/Users/David/Desktop/GitLab/sara-build-vs2022-static/frag.spv";
#else
      static const auto vs_path =
          "/home/david/GitLab/oddkiva/sara-build-Asan/vert.spv";
      static const auto fs_path =
          "/home/david/GitLab/oddkiva/sara-build-Asan/frag.spv";
#endif

      _graphics_pipeline =
          GraphicsPipeline::Builder{_device, _render_pass}
              .vertex_shader_path(vs_path)
              .fragment_shader_path(fs_path)
              .vbo_data_format<Vertex>()
              .input_assembly_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
              .viewport_sizes(static_cast<float>(w), static_cast<float>(h))
              .scissor_sizes(w, h)
              .create();
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
    Shakti::Vulkan::Queue _graphics_queue;
    Shakti::Vulkan::Queue _present_queue;

    // The abstraction of the present operations in the hardware
    Swapchain _swapchain;
    RenderPass _render_pass;

    GraphicsPipeline _graphics_pipeline;

    // The draw command machinery
    Shakti::Vulkan::CommandPool _command_pool;
    Shakti::Vulkan::CommandBufferSequence _command_buffers;

    // Synchronization objects.
    std::vector<Shakti::Vulkan::Fence> _fences;
    std::vector<Shakti::Vulkan::Semaphore> _semaphores;
  };

}  // namespace DO::Kalpana::Vulkan
