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

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <DO/Shakti/Vulkan/GraphicsBackend.hpp>
#include <DO/Shakti/Vulkan/Semaphore.hpp>


using namespace DO::Kalpana::Vulkan;


auto GraphicsBackend::init_instance(const std::string& app_name,
                                    const bool debug_vulkan) -> void
{
  // Vulkan instance.
  _instance_extensions = Surface::list_required_instance_extensions_from_glfw();
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

auto GraphicsBackend::init_surface(GLFWwindow* window) -> void
{
  _surface = Surface{_instance, window};
}

auto GraphicsBackend::init_physical_device() -> void
{
  namespace svk = Shakti::Vulkan;

  // List all Vulkan physical devices.
  const auto physical_devices =
      svk::PhysicalDevice::list_physical_devices(_instance);

  // Find a suitable physical (GPU) device that can be used for 3D graphics
  // application.
  const auto di = std::find_if(
      physical_devices.begin(), physical_devices.end(), [this](const auto& d) {
        return d.supports_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME) &&
               !find_graphics_queue_family_indices(d).empty() &&
               !find_present_queue_family_indices(d, _surface).empty();
      });
  if (di == physical_devices.end())
  {
    std::cerr << "[VK] Error: could not find suitable Vulkan physical device!";
    return;
  }

  // There must be a suitable GPU device...
  _physical_device = *di;
}

auto GraphicsBackend::init_device_and_queues() -> void
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
                .enable_queue_families(
                    {graphics_queue_family_index, present_queue_family_index})
                .enable_physical_device_features({})
                .enable_validation_layers(_validation_layers)
                .create();

  SARA_DEBUG
      << "[VK] - Fetching the graphics queue from the logical device...\n";
  _graphics_queue = svk::Queue{_device, graphics_queue_family_index};
  SARA_DEBUG
      << "[VK] - Fetching the present queue from the logical device...\n";
  _present_queue = svk::Queue{_device, present_queue_family_index};
}

auto GraphicsBackend::init_swapchain(GLFWwindow* window) -> void
{
  _swapchain = Swapchain{_physical_device, _device, _surface, window};
}

auto GraphicsBackend::init_swapchain_fbos() -> void
{
  _swapchain_fbos = FramebufferSequence{_swapchain, _render_pass};
}

auto GraphicsBackend::init_render_pass() -> void
{
  _render_pass.create_basic_render_pass(_device, _swapchain.image_format);
}

auto GraphicsBackend::init_command_pool_and_buffers() -> void
{
  namespace svk = Shakti::Vulkan;

  const auto graphics_queue_family_index =
      find_graphics_queue_family_indices(_physical_device).front();
  SARA_CHECK(graphics_queue_family_index);

  _graphics_cmd_pool = svk::CommandPool{_device, graphics_queue_family_index};
  SARA_DEBUG << fmt::format(
      "[VK] Initialized command pool {}\n",
      fmt::ptr(static_cast<VkCommandPool>(_graphics_cmd_pool)));

  _graphics_cmd_bufs = svk::CommandBufferSequence{
      static_cast<std::uint32_t>(_swapchain.images.size()),  //
      _device,                                               //
      _graphics_cmd_pool                                     //
  };
}

auto GraphicsBackend::init_synchronization_objects() -> void
{
  namespace svk = Shakti::Vulkan;

  // Create as many synchronization objects as swapchain images.
  _render_fences.resize(_swapchain.images.size());
  // Initialize them with an unsignaled state.
  for (auto& fence : _render_fences)
    fence = svk::Fence{_device};

  _image_available_semaphores.resize(_swapchain.images.size());
  for (auto& s : _image_available_semaphores)
    s = svk::Semaphore{_device};

  _render_finished_semaphores.resize(_swapchain.images.size());
  for (auto& s : _render_finished_semaphores)
    s = svk::Semaphore{_device};
}
