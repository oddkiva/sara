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

#pragma once

#include <DO/Kalpana/Math/Projection.hpp>

#include <Eigen/Geometry>


// The explorer's eye.
struct Camera
{
  // Default camera values
  static constexpr auto YAW = -90.0f;
  static constexpr auto PITCH = 0.0f;
  static constexpr auto SPEED = 1e-1f;
  static constexpr auto SENSITIVITY = 1e-1f;
  static constexpr auto ZOOM = 45.0f;

  Eigen::Vector3f position{10.f * Eigen::Vector3f::UnitY()};
  Eigen::Vector3f front{-Eigen::Vector3f::UnitZ()};
  Eigen::Vector3f up{Eigen::Vector3f::UnitY()};
  Eigen::Vector3f right;
  Eigen::Vector3f world_up{Eigen::Vector3f::UnitY()};

  float yaw{YAW};
  float pitch{PITCH};
  float roll{0.f};

  float movement_speed{SPEED};
  float movement_sensitivity{SENSITIVITY};
  float zoom{ZOOM};

  auto move_left(float delta)
  {
    position -= movement_speed * delta * right;
  }

  auto move_right(float delta)
  {
    position += movement_speed * delta * right;
  }

  auto move_forward(float delta)
  {
    position += movement_speed * delta * front;
  }

  auto move_backward(float delta)
  {
    position -= movement_speed * delta * front;
  }

  auto move_up(float delta)
  {
    position += movement_speed * delta * up;
  }

  auto move_down(float delta)
  {
    position -= movement_speed * delta * up;
  }

  // pitch
  auto yes_head_movement(float delta)
  {
    pitch += movement_sensitivity * delta;
  }

  // yaw
  auto no_head_movement(float delta)
  {
    yaw += movement_sensitivity * delta;
  }

  auto maybe_head_movement(float delta)
  {
    roll += movement_sensitivity * delta;
  }

  auto update()
  {
    Eigen::Vector3f front1;

    static constexpr auto pi = static_cast<float>(M_PI);
    front1 << cos(yaw * pi / 180) * cos(pitch * pi / 180.f),
        sin(pitch * pi / 180.f),
        sin(yaw * pi / 180.f) * cos(pitch * pi / 180.f);
    front = front1.normalized();

    right = front.cross(world_up).normalized();
    right =
        Eigen::AngleAxisf(roll * pi / 180, front).toRotationMatrix() * right;
    right.normalize();

    up = right.cross(front).normalized();
  }

  auto view_matrix() -> Eigen::Matrix4f
  {
    namespace k = DO::Kalpana;
    return k::look_at(position, (position + front).eval(), up);
  }
};
