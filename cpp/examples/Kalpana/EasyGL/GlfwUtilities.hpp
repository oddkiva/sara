// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <GLFW/glfw3.h>

#include "Camera.hpp"
#include "Time.hpp"


inline auto move_camera_from_keyboard(GLFWwindow* window, Camera& camera,
                                      Time& time) -> void
{
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.move_forward(time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.move_backward(time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.move_left(time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.move_right(time.delta_time);

  if (glfwGetKey(window, GLFW_KEY_DELETE) == GLFW_PRESS)
    camera.no_head_movement(-time.delta_time);  // CCW
  if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
    camera.no_head_movement(+time.delta_time);  // CW

  if (glfwGetKey(window, GLFW_KEY_HOME) == GLFW_PRESS)
    camera.yes_head_movement(+time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_END) == GLFW_PRESS)
    camera.yes_head_movement(-time.delta_time);

  if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    camera.move_up(time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
    camera.move_down(time.delta_time);

  if (glfwGetKey(window, GLFW_KEY_INSERT) == GLFW_PRESS)
    camera.maybe_head_movement(-time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS)
    camera.maybe_head_movement(+time.delta_time);

  camera.update();
}
