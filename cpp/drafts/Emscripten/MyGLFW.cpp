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

#include "MyGLFW.hpp"

#include <iostream>


GLFWwindow* MyGLFW::window = nullptr;
int MyGLFW::width = -1;
int MyGLFW::height = -1;


auto MyGLFW::initialize() -> bool
{
  if (glfwInit() != GL_TRUE)
  {
    std::cout << "Failed to initialize GLFW!" << std::endl;
    glfwTerminate();
    return false;
  }

  window = glfwCreateWindow(512, 512, "OpenGL Window", NULL, NULL);
  if (!MyGLFW::window)
  {
    std::cout << "Failed to create window!" << std::endl;
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window);

  // Set the appropriate mouse and keyboard callbacks.
  glfwGetFramebufferSize(window, &width, &height);
  glfwSetWindowSizeCallback(window, window_size_callback);
  glfwSetMouseButtonCallback(window, mouse_callback);
  glfwSetKeyCallback(window, key_callback);

  std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;

  return true;
}

auto MyGLFW::window_size_callback(GLFWwindow* /* window */, int width,
                                  int height) -> void
{
  std::cout << "window_size_callback received width: " << width
            << "  height: " << height << std::endl;
}

auto MyGLFW::key_callback(GLFWwindow* /* window */, int key, int /* scancode */,
                          int action, int /* modifier */) -> void
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
    glfwSetWindowShouldClose(window, 1);

  if (key == GLFW_KEY_ENTER)
    std::cout << "Hit Enter!" << std::endl;
}

auto MyGLFW::mouse_callback(GLFWwindow* /* window */, int button,
                            int /* action */, int /* modifiers */) -> void
{
  std::cout << "Clicked mouse button: " << button << "!" << std::endl;
}
