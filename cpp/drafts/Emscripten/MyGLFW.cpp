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
#include "MetricGridRenderer.hpp"
#include "Scene.hpp"

#include "MyGLFW.hpp"

#include <DO/Sara/Core/Math/Rotation.hpp>

#include <Eigen/Core>

#include <iostream>


GLFWwindow* MyGLFW::window = nullptr;
int MyGLFW::width = -1;
int MyGLFW::height = -1;
int MyGLFW::high_dpi_scale_factor = 1.0f;
std::string MyGLFW::glsl_version = "#version 300 es";


auto MyGLFW::initialize(int width, int height) -> bool
{
  if (glfwInit() != GL_TRUE)
  {
    std::cout << "Failed to initialize GLFW!" << std::endl;
    glfwTerminate();
    return false;
  }

  window = glfwCreateWindow(width, height, "OpenGL Window", NULL, NULL);
  if (!MyGLFW::window)
  {
    std::cout << "Failed to create window!" << std::endl;
    glfwTerminate();
    return false;
  }

  MyGLFW::width = width;
  MyGLFW::height = height;


// #ifdef __APPLE__
//   // GL 3.2 + GLSL 150
//   MyGLFW::glsl_version = "#version 150";
//   glfwWindowHint(  // required on Mac OS
//       GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
// #elif __linux__
//   // GL 3.2 + GLSL 150
//   glsl_version = "#version 150";
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
// #elif _WIN32
//   // GL 3.0 + GLSL 130
//   glsl_version = "#version 130";
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
// #endif

#ifdef _WIN32
  // if it's a HighDPI monitor, try to scale everything
  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  float xscale, yscale;
  glfwGetMonitorContentScale(monitor, &xscale, &yscale);
  if (xscale > 1 || yscale > 1)
  {
    high_dpi_scale_factor = static_cast<int>(xscale);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
  }
#elif __APPLE__
  // to prevent 1200x800 from becoming 2400x1600
  glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif


  glfwMakeContextCurrent(window);

  // Set the appropriate mouse and keyboard callbacks.
  glfwGetFramebufferSize(window, &MyGLFW::width, &MyGLFW::height);
  glfwSetWindowSizeCallback(window, window_size_callback);
  glfwSetMouseButtonCallback(window, mouse_callback);
  glfwSetKeyCallback(window, key_callback);
  glfwSetScrollCallback(window, scroll_callback);

  return true;
}

auto MyGLFW::window_size_callback(GLFWwindow* /* window */, int width,
                                  int height) -> void
{
  MyGLFW::width = width;
  MyGLFW::height = height;

  auto& scene = Scene::instance();

  const auto aspect_ratio = static_cast<float>(width) / height;
  scene._projection = orthographic(-0.5f * aspect_ratio, 0.5f * aspect_ratio,
                                   -0.5f, 0.5f, -0.5f, 0.5f);
}

auto MyGLFW::key_callback(GLFWwindow* /* window */, int key, int /* scancode */,
                          int action, int /* modifier */) -> void
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
    glfwSetWindowShouldClose(window, 1);

  auto& scene = Scene::instance();
  switch (key)
  {
  case GLFW_KEY_LEFT:
    scene._model_view(0, 3) += 0.01f;
    break;
  case GLFW_KEY_RIGHT:
    scene._model_view(0, 3) -= 0.01f;
    break;
  case GLFW_KEY_UP:
    scene._model_view(1, 3) += 0.01f;
    break;
  case GLFW_KEY_DOWN:
    scene._model_view(1, 3) -= 0.01f;
    break;
  default:
    break;
  };
}

void MyGLFW::scroll_callback(GLFWwindow* /*window*/, double /*xoffset */,
                             double yoffset)
{
  auto& scene = Scene::instance();
  if (yoffset > 0)
  {
    scene._model_view.topLeftCorner(3, 4) *= 1.05f;
    return;
  }
  if (yoffset < 0)
  {
    scene._model_view.topLeftCorner(3, 4) /= 1.05f;
    return;
  }
}


auto MyGLFW::mouse_callback(GLFWwindow* /* window */, int button,
                            int /* action */, int /* modifiers */) -> void
{
  std::cout << "Clicked mouse button: " << button << "!" << std::endl;
}
