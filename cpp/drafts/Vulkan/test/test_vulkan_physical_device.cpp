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
#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>

#include <boost/test/unit_test.hpp>


static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
static constexpr auto compiling_for_apple = true;
#else
static constexpr auto compiling_for_apple = false;
#endif


BOOST_AUTO_TEST_CASE(test_find_physical_devices_for_3d_graphics_application)
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

  // Vulkan surface.
  auto surface = kvk::Surface{};
  surface.init(instance, window);

  // Vulkan physical device.
  const auto physical_devices =
      svk::PhysicalDevice::list_physical_devices(instance);

  // All my physical devices are GPU devices (so far...).
  //
  // - All of them should support the graphics swapchain.
  BOOST_CHECK(std::all_of(
      physical_devices.begin(), physical_devices.end(), [](const auto& d) {
        return d.supports_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
      }));
  // All my GPUs should support the graphics queue family.
  auto gpu_id = 0;
  for (const auto& d : physical_devices)
  {
    const auto graphics_queue_family_indices =
        kvk::find_graphics_queue_family_indices(d);

    SARA_DEBUG << fmt::format("[GPU {}] Graphics queue family indices:\n",
                              gpu_id);
    for (const auto i : graphics_queue_family_indices)
      SARA_DEBUG << fmt::format("- {}\n", i);
    BOOST_CHECK(!graphics_queue_family_indices.empty());

    ++gpu_id;
  }

  // All my GPUs should support both the present queue family.
  //
  // But to my understanding, not all of my GPUs can operate on the Vulkan
  // surface I just created.
  gpu_id = 0;
  auto one_gpu_can_present_on_the_vulkan_surface = false;
  for (const auto& d : physical_devices)
  {
    const auto present_queue_family_indices =
        kvk::find_present_queue_family_indices(d, surface);

    SARA_DEBUG << fmt::format("[GPU {}] Present queue family indices:\n",
                              gpu_id);
    for (const auto i : present_queue_family_indices)
      SARA_DEBUG << fmt::format("- {}\n", i);
    BOOST_CHECK(!present_queue_family_indices.empty());

    if (!present_queue_family_indices.empty())
    {
      one_gpu_can_present_on_the_vulkan_surface = true;
      break;
    }

    ++gpu_id;
  }
  BOOST_CHECK(one_gpu_can_present_on_the_vulkan_surface);


  surface.destroy(instance);

  if (window != nullptr)
    glfwDestroyWindow(window);
}
