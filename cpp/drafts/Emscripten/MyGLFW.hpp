// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <GLFW/glfw3.h>

#include <string>


struct MyGLFW
{
  static GLFWwindow* window;
  static int width;
  static int height;
  static int high_dpi_scale_factor;
  static std::string glsl_version;

  static auto initialize(int width = 1200, int height = 540) -> bool;

  static void window_size_callback(GLFWwindow* /* window */, int width,
                                   int height);

  static void key_callback(GLFWwindow* /* window */, int key,
                           int /* scancode */, int action, int /* modifier */);

  static void scroll_callback(GLFWwindow* /*window*/, double /*xoffset */,
                              double yoffset);

  static void mouse_callback(GLFWwindow* /* window */, int button,
                             int /* action */, int /* modifiers */);
};
