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

#define BOOST_TEST_MODULE "Vulkan/Graphics Backend"
#define GLFW_INCLUDE_VULKAN

#include <drafts/Vulkan/EasyGLFW.hpp>
#include <drafts/Vulkan/GraphicsBackend.hpp>

#include <DO/Sara/Defines.hpp>

#include <boost/test/unit_test.hpp>


static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
static constexpr auto compile_for_apple = true;
#else
static constexpr auto compile_for_apple = false;
#endif


BOOST_AUTO_TEST_CASE(test_graphics_backend)
{
  namespace glfw = DO::Kalpana::GLFW;
  namespace kvk = DO::Kalpana::Vulkan;

  auto glfw_app = glfw::Application{};
  glfw_app.init_for_vulkan_rendering();

  // Create a window.
  const auto window = glfw::Window(100, 100, "Vulkan");

  auto vk_backend = kvk::GraphicsBackend{window, "GLFW-Vulkan App", true};
}

