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

#define BOOST_TEST_MODULE "Vulkan/Instance"
#define GLFW_INCLUDE_VULKAN

#include <DO/Shakti/Vulkan/EasyGLFW.hpp>
#include <DO/Shakti/Vulkan/Instance.hpp>
#include <DO/Shakti/Vulkan/Surface.hpp>

#include <GLFW/glfw3.h>

#include <boost/test/unit_test.hpp>


#if defined(__APPLE__)
static constexpr auto compiling_for_apple = true;
#else
static constexpr auto compiling_for_apple = false;
#endif


BOOST_AUTO_TEST_CASE(test_barebone_instance)
{
  namespace svk = DO::Shakti::Vulkan;

  const auto instance_extensions =
      compiling_for_apple ? std::vector{"VK_KHR_portability_enumeration"}
                          : std::vector<const char*>{};

  const auto instance = svk::Instance::Builder{}
                            .application_name("Barebone Vulkan Application")
                            .engine_name("No Engine")
                            .enable_instance_extensions(instance_extensions)
                            .create();
}

BOOST_AUTO_TEST_CASE(test_glfw_vulkan_instance)
{
  namespace svk = DO::Shakti::Vulkan;
  namespace kvk = DO::Kalpana::Vulkan;

  static constexpr auto debug_vulkan_instance = true;

  glfwInit();

  auto instance_extensions =
      kvk::Surface::list_required_instance_extensions_from_glfw();
  if constexpr (debug_vulkan_instance)
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  if constexpr (compiling_for_apple)
    instance_extensions.emplace_back("VK_KHR_portability_enumeration");

  SARA_DEBUG << "Inspecting all required Vulkan extensions:\n";
  for (const auto extension : instance_extensions)
    SARA_DEBUG << fmt::format("- {}\n", extension);

  const auto validation_layers =
      debug_vulkan_instance ? std::vector{"VK_LAYER_KHRONOS_validation"}
                            : std::vector<const char*>{};

  const auto instance = svk::Instance::Builder{}
                            .application_name("GLFW-Vulkan Application")
                            .engine_name("No Engine")
                            .enable_instance_extensions(instance_extensions)
                            .enable_validation_layers(validation_layers)
                            .create();

  glfwTerminate();
}
