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


BOOST_AUTO_TEST_CASE(test_graphics_backend)
{
  namespace glfw = DO::Kalpana::GLFW;
  namespace kvk = DO::Kalpana::Vulkan;

  auto glfw_app = glfw::Application{};
  glfw_app.init_for_vulkan_rendering();

  // Create a window.
  const auto window = glfw::Window(100, 100, "Vulkan");

  namespace fs = std::filesystem;
  static const auto program_path =
      fs::path(boost::unit_test::framework::master_test_suite().argv[0]);
  static const auto shader_dir_path =
      program_path.parent_path() / "test_shaders";
  static const auto vshader_path = shader_dir_path / "vert.spv";
  static const auto fshader_path = shader_dir_path / "frag.spv";

  static constexpr auto with_vulkan_logging = true;

  const auto vk_backend =
      kvk::GraphicsBackend{window, "GLFW-Vulkan App", vshader_path,
                           fshader_path, with_vulkan_logging};
}
