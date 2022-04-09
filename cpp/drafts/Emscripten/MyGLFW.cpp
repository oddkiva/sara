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

// Include this first before GLFW otherwise the compilation will fail.
#include "Geometry.hpp"
#include "Scene.hpp"

#include "MyGLFW.hpp"

#include <Eigen/Core>

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

  window = glfwCreateWindow(1024, 1024, "OpenGL Window", NULL, NULL);
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

  return true;
}

auto MyGLFW::window_size_callback(GLFWwindow* /* window */, int width,
                                  int height) -> void
{
  MyGLFW::width = width;
  MyGLFW::height = height;

  auto& scene = Scene::instance();

  const auto aspect_ratio = static_cast<float>(width) / height;
  scene._projection = orthographic(-0.5f * aspect_ratio, 0.5f * aspect_ratio, -0.5f, 0.5f, -0.5f, 0.5f);
}

auto MyGLFW::key_callback(GLFWwindow* /* window */, int key, int /* scancode */,
                          int action, int /* modifier */) -> void
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
    glfwSetWindowShouldClose(window, 1);

  auto& scene = Scene::instance();
  switch (key) {
  case GLFW_KEY_A:
    scene._model_view(0, 3) += 0.01f;
    break;
  case GLFW_KEY_D:
    scene._model_view(0, 3) -= 0.01f;
    break;
  case GLFW_KEY_W:
    scene._model_view(1, 3) += 0.01f;
    break;
  case GLFW_KEY_S:
    scene._model_view(1, 3) -= 0.01f;
    break;
  case GLFW_KEY_R:
    scene._model_view.topLeftCorner(3, 4) *= 1.01f;
    break;
  case GLFW_KEY_F:
    scene._model_view.topLeftCorner(3, 4) /= 1.01f;
    break;
  default:
    break;
  };
}

auto MyGLFW::mouse_callback(GLFWwindow* /* window */, int button,
                            int /* action */, int /* modifiers */) -> void
{
  std::cout << "Clicked mouse button: " << button << "!" << std::endl;
}
