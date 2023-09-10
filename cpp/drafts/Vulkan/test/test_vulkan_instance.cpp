// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "EasyVulkan/Vulkan Instance"

#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/VulkanGLFWInterop.hpp>

#include <GLFW/glfw3.h>

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(test_graphics_instance)
{
  namespace svk = DO::Shakti::EasyVulkan;
  namespace k = DO::Kalpana;


  static constexpr auto debug_vulkan_instance = true;

  glfwInit();

  auto instance_extensions = k::list_required_vulkan_extensions_from_glfw();
  if constexpr (debug_vulkan_instance)
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  const auto validation_layers_required =
      debug_vulkan_instance ? std::vector{"VK_LAYER_KHRONOS_validation"}
                            : std::vector<const char*>{};

  auto instance = svk::InstanceCreator{}
                      .application_name("GLFW-Vulkan Application")
                      .engine_name("No Engine")
                      .required_instance_extensions(instance_extensions)
                      .required_validation_layers(validation_layers_required)
                      .create();

  glfwTerminate();
}
