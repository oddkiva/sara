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

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(test_instance_constructor)
{
  namespace svk = DO::Shakti::EasyVulkan;
  namespace k = DO::Kalpana;

  // const auto extensions_required_by_glfw =
  //     k::list_required_vulkan_extensions_from_glfw(true);
  // const auto validation_layers_required = std::vector{
  //     "VK_LAYER_KHRONOS_validation"  //
  // };
  const auto extensions_required_by_glfw =
      k::list_required_vulkan_extensions_from_glfw(false);
  const auto validation_layers_required = std::vector<const char*>{};

  auto instance = svk::Instance{};

  VkInstance& vk_instance = instance;

  svk::InstanceCreator{}
      // .application_name("GLFW-Vulkan Application")
      // .engine_name("No Engine")
      // .required_instance_extensions(extensions_required_by_glfw)
      // .required_validation_layers(validation_layers_required)
      .init(vk_instance);
}
