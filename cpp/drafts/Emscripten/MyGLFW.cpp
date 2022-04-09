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
  auto& grid_renderer = MetricGridRenderer::instance();
  static auto yaw_pitch_roll = std::array<float, 3>{0, 0, 0};
  static auto rotation_changed = false;
  // clang-format off
  static const Eigen::Matrix3f P = (Eigen::Matrix3f{} <<
     0,  0, 1, // Camera Z =          Automotive X
    -1,  0, 0, // Camera X = Negative Automotive Y
     0, -1, 0  // Camera Y = Negative Automotive Z
  ).finished();
  // clang-format on


  static constexpr auto angle_step = 0.5f * static_cast<float>(M_PI) / 180;

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
  case GLFW_KEY_A:
    yaw_pitch_roll[0] += angle_step;
    rotation_changed = true;
    break;
  case GLFW_KEY_D:
    yaw_pitch_roll[0] -= angle_step;
    rotation_changed = true;
    break;
  case GLFW_KEY_W:
    yaw_pitch_roll[1] += angle_step;
    rotation_changed = true;
    break;
  case GLFW_KEY_S:
    yaw_pitch_roll[1] -= angle_step;
    rotation_changed = true;
    break;
  case GLFW_KEY_Q:
    yaw_pitch_roll[2] += angle_step;
    rotation_changed = true;
    break;
  case GLFW_KEY_E:
    yaw_pitch_roll[2] -= angle_step;
    rotation_changed = true;
    break;
  default:
    break;
  };

  if (rotation_changed)
  {
    const auto R = DO::Sara::rotation(yaw_pitch_roll[0], yaw_pitch_roll[1],
                                      yaw_pitch_roll[2]) *
                   P;
    const auto t = Eigen::Vector3f(0, 0, 1.51);
    grid_renderer._extrinsics.topLeftCorner(3, 3) = R.transpose();
    grid_renderer._extrinsics.block<3, 1>(0, 3) = -R.transpose() * t;
    rotation_changed = false;
  }
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
