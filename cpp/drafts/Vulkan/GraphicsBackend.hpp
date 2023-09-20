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
#include <drafts/Vulkan/Framebuffer.hpp>
#include <drafts/Vulkan/GraphicsPipeline.hpp>
#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>
#include <drafts/Vulkan/Queue.hpp>
#include <drafts/Vulkan/RenderPass.hpp>
#include <drafts/Vulkan/Semaphore.hpp>
#include <drafts/Vulkan/Surface.hpp>
#include <drafts/Vulkan/Swapchain.hpp>


namespace DO::Kalpana::Vulkan {

  class GraphicsBackend
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
    GraphicsBackend(GLFWwindow* window, const std::string& app_name,
                    const bool debug_vulkan);

    auto init_instance(const std::string& app_name, const bool debug_vulkan)
        -> void;

    auto init_surface(GLFWwindow* window) -> void;

    auto init_physical_device() -> void;

    auto init_device_and_queues() -> void;

    auto init_swapchain(GLFWwindow* window) -> void;

    auto init_framebuffers() -> void;

    auto init_render_pass() -> void;

    auto init_graphics_pipeline(GLFWwindow* window) -> void;

    auto init_command_pool_and_buffers() -> void;

    auto init_synchronization_objects() -> void;

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
    // Color buffers, render subpasses...
    RenderPass _render_pass;
    // Framebuffers are wrapped swapchain image views... (somehow)
    // and associated to a render pass.
    FramebufferSequence _framebuffers;

    GraphicsPipeline _graphics_pipeline;

    // The draw command machinery
    Shakti::Vulkan::CommandPool _graphics_cmd_pool;
    Shakti::Vulkan::CommandBufferSequence _graphics_cmd_bufs;

    // Synchronization objects.
    std::vector<Shakti::Vulkan::Fence> _render_fences;
    std::vector<Shakti::Vulkan::Semaphore> _image_available_semaphores;
    std::vector<Shakti::Vulkan::Semaphore> _render_finished_semaphores;
  };

}  // namespace DO::Kalpana::Vulkan
