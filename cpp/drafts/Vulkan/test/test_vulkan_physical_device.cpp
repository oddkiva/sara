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

#include <drafts/Vulkan/GLFWHelpers.hpp>
#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>

#include <boost/test/unit_test.hpp>


static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
static constexpr auto compiling_for_apple = true;
#else
static constexpr auto compiling_for_apple = false;
#endif


BOOST_AUTO_TEST_CASE(test_list_physical_devices)
{
  namespace svk = DO::Shakti::Vulkan;
  namespace k = DO::Kalpana;
  namespace kvk = DO::Kalpana::Vulkan;

  auto glfw_app = DO::Kalpana::GLFWApplication{};
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

  // All my devices should support graphics operations.
  for (const auto& physical_device : physical_devices)
  {
    const auto& queue_families = physical_device.queue_families;
    SARA_CHECK(queue_families.size());
  }

  surface.destroy(instance);

  if (window != nullptr)
    glfwDestroyWindow(window);
}
