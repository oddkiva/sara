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

#define BOOST_TEST_MODULE "EasyVulkan/Vulkan Physical Device"

#include <drafts/Vulkan/VulkanGLFWInterop.hpp>

#include <drafts/Vulkan/GLFWApplication.hpp>

#include <drafts/Vulkan/Device.hpp>
#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>

#include <boost/test/unit_test.hpp>


using std::find_if;

static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
static constexpr auto compiling_for_apple = true;
#else
static constexpr auto compiling_for_apple = false;
#endif


BOOST_AUTO_TEST_CASE(test_device)
{
  namespace svk = DO::Shakti::Vulkan;
  namespace k = DO::Kalpana;
  namespace kvk = DO::Kalpana::Vulkan;

  auto glfw_app = k::GLFWApplication{};
  glfw_app.init_for_vulkan_rendering();

  // Create a window.
  const auto window = glfwCreateWindow(100, 100,  //
                                       "Vulkan",  //
                                       nullptr, nullptr);

  // Vulkan instance.
  auto instance_extensions = kvk::list_required_vulkan_extensions_from_glfw();
  if constexpr (debug_vulkan_instance)
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  if constexpr (compiling_for_apple)
    instance_extensions.push_back("VK_KHR_portability_enumeration");

  const auto validation_layers_required =
      debug_vulkan_instance ? std::vector{"VK_LAYER_KHRONOS_validation"}
                            : std::vector<const char*>{};

  const auto instance =
      svk::InstanceCreator{}
          .application_name("GLFW-Vulkan Application")
          .engine_name("No Engine")
          .required_instance_extensions(instance_extensions)
          .required_validation_layers(validation_layers_required)
          .create();

  // Initialize a Vulkan surface to which the GLFW Window surface is bound.
  auto surface = kvk::Surface{};
  surface.init(instance, window);

  // List all Vulkan physical devices.
  const auto physical_devices =
      svk::PhysicalDevice::list_physical_devices(instance);

  // Find a suitable physical (GPU) device that can be used for 3D graphics
  // application.
  const auto di = find_if(
      physical_devices.begin(), physical_devices.end(),
      [&surface](const auto& d) {
        return d.supports_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME) &&
               !kvk::find_graphics_queue_family_indices(d).empty() &&
               !kvk::find_present_queue_family_indices(d, surface).empty();
      });

  // There must be a suitable GPU device...
  BOOST_CHECK(di != physical_devices.end());
  const auto& physical_device = *di;

  // According to:
  // https://stackoverflow.com/questions/61434615/in-vulkan-is-it-beneficial-for-the-graphics-queue-family-to-be-separate-from-th
  //
  // Using distinct queue families, namely one for the graphics operations and
  // another for the present operations, does not result in better performance.
  //
  // This is because the hardware does not expose present-only queue families...
  const auto graphics_queue_family_index =
      kvk::find_graphics_queue_family_indices(physical_device).front();
  const auto present_queue_family_index =
      kvk::find_present_queue_family_indices(physical_device, surface).front();

  // Create a logical device.
  const auto device =
      svk::DeviceCreator{*di}
          .enable_device_extensions({VK_KHR_SWAPCHAIN_EXTENSION_NAME})
          .enable_queue_families(
              {graphics_queue_family_index, present_queue_family_index})
          .enable_device_features({})
          .enable_validation_layers(validation_layers_required)
          .create();
  BOOST_CHECK(device.handle != nullptr);

  // Destroy in this order.
  surface.destroy(instance);
  if (window != nullptr)
    glfwDestroyWindow(window);
}
