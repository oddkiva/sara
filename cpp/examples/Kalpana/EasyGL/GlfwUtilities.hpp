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

#include <DO/Kalpana/EasyGL/TrackBall.hpp>

#include "Camera.hpp"
#include "Time.hpp"

#include <iostream>


auto camera = Camera{};
auto gtime = Time{};

auto trackball = DO::Kalpana::GL::TrackBall{};
Eigen::Quaternionf q = Eigen::Quaternionf::Identity();

auto show_checkerboard = true;
auto scale = 1.f;


static inline auto normalize_cursor_pos(GLFWwindow* window,
                                        const Eigen::Vector2d& pos)
    -> Eigen::Vector2d
{
  auto w = int{};
  auto h = int{};
  glfwGetWindowSize(window, &w, &h);

  const Eigen::Vector2d c = Eigen::Vector2i(w, h).cast<double>() * 0.5;

  Eigen::Vector2d normalized_pos = ((pos - c).array() / c.array()).matrix();
  normalized_pos.y() *= -1;
  return normalized_pos;
};


inline auto move_camera_from_keyboard(GLFWwindow* window,  //
                                      int key,             //
                                      int /* scancode */,  //
                                      int action,          //
                                      int /* mods */) -> void
{
  if (action == GLFW_RELEASE)
    return;

  if (key == GLFW_KEY_W)
    camera.move_forward(gtime.delta_time);
  if (key == GLFW_KEY_S)
    camera.move_backward(gtime.delta_time);
  if (key == GLFW_KEY_A)
    camera.move_left(gtime.delta_time);
  if (key == GLFW_KEY_D)
    camera.move_right(gtime.delta_time);

  if (key == GLFW_KEY_DELETE)
    camera.no_head_movement(-gtime.delta_time);  // CCW
  if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
    camera.no_head_movement(+gtime.delta_time);  // CW

  if (key == GLFW_KEY_HOME)
    camera.yes_head_movement(+gtime.delta_time);
  if (key == GLFW_KEY_END)
    camera.yes_head_movement(-gtime.delta_time);

  if (key == GLFW_KEY_R)
    camera.move_up(gtime.delta_time);
  if (key == GLFW_KEY_F)
    camera.move_down(gtime.delta_time);

  if (key == GLFW_KEY_INSERT)
    camera.maybe_head_movement(-gtime.delta_time);
  if (key == GLFW_KEY_PAGE_UP)
    camera.maybe_head_movement(+gtime.delta_time);

  if (key == GLFW_KEY_SPACE)
    show_checkerboard = !show_checkerboard;

  if (key == GLFW_KEY_MINUS)
    scale /= 1.01f;
  if (key == GLFW_KEY_EQUAL)
    scale *= 1.11f;

  camera.update();
}


inline auto use_trackball(GLFWwindow* window, int button, int action, int /* mods */)
{
  if (button != GLFW_MOUSE_BUTTON_LEFT)
    return;

  auto p = Eigen::Vector2d{};
  glfwGetCursorPos(window, &p.x(), &p.y());

  const Eigen::Vector2f pf = normalize_cursor_pos(window, p).cast<float>();
  if (action == GLFW_PRESS && !trackball.pressed())
    trackball.push(pf);
  else if (action == GLFW_RELEASE && trackball.pressed())
    trackball.release(pf);
}

inline auto move_trackball(GLFWwindow* window, double x, double y) -> void
{
  const auto curr_pos = Eigen::Vector2d{x, y};
  const Eigen::Vector2f p =
      normalize_cursor_pos(window, curr_pos).cast<float>();

  if (trackball.pressed())
    trackball.move(p);
}
