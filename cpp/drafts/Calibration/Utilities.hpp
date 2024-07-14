// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <drafts/Calibration/Chessboard.hpp>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/OmnidirectionalCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/PnP/LambdaTwist.hpp>


namespace DO::Sara {

  inline auto init_calibration_matrix(v2::OmnidirectionalCamera<double>& camera,
                                      const int w, const int h) -> void
  {
    const auto d = static_cast<double>(std::max(w, h));
    const auto f = 0.5 * d;

    camera.fx() = f;
    camera.fy() = f;
    camera.shear() = 0;
    camera.u0() = w * 0.5;
    camera.v0() = w * 0.5;
  }

  inline auto estimate_pose_with_p3p(const ChessboardCorners& cb,
                                     const v2::OmnidirectionalCamera<double>& K)
      -> std::optional<Eigen::Matrix<double, 3, 4>>
  {
    auto points = Eigen::Matrix3d{};
    auto rays = Eigen::Matrix3d{};

    SARA_DEBUG << "Filling points and rays for P3P..." << std::endl;
    auto xs = std::array{0, 1, 0};
    auto ys = std::array{0, 0, 1};
    for (auto n = 0; n < 3; ++n)
    {
      const auto& x = xs[n];
      const auto& y = ys[n];
      const Eigen::Vector2d xn = cb.image_point(x, y).cast<double>();
      if (is_nan(xn))
        continue;

      points.col(n) = cb.scene_point(x, y);
      rays.col(n) = K.backproject(xn).normalized();
    }
    if (is_nan(points) || is_nan(rays))
      return std::nullopt;

    SARA_DEBUG << "Solving P3P..." << std::endl;
    SARA_DEBUG << "Points =\n" << points << std::endl;
    SARA_DEBUG << "Rays   =\n" << rays << std::endl;
    const auto poses = solve_p3p(points, rays);
    if (poses.empty())
      return std::nullopt;

    // Find the best poses.
    SARA_DEBUG << "Determining the best pose..." << std::endl;
    auto errors = std::vector<double>{};
    for (const auto& pose : poses)
    {
      auto error = 0;

      auto n = 0;
      for (auto y = 0; y < cb.height(); ++y)
      {
        for (auto x = 0; x < cb.width(); ++x)
        {
          auto x0 = cb.image_point(x, y);
          if (is_nan(x0))
            continue;

          const auto& R = pose.topLeftCorner<3, 3>();
          const auto& t = pose.col(3);
          const auto X = (R * cb.scene_point(x, y) + t);


          const Eigen::Vector2f x1 = K.project(X).cast<float>();
          error += (x1 - x0).squaredNorm();
          ++n;
        }
      }

      errors.emplace_back(error / n);
    }

    const auto best_pose_index =
        std::min_element(errors.begin(), errors.end()) - errors.begin();
    const auto& best_pose = poses[best_pose_index];
    SARA_DEBUG << "Best pose:\n" << best_pose << std::endl;

    return best_pose;
  }

  inline auto estimate_pose_with_p3p(const ChessboardCorners& cb,
                                     const Eigen::Matrix3d& K)
      -> std::optional<Eigen::Matrix<double, 3, 4>>
  {
    auto points = Eigen::Matrix3d{};
    auto rays = Eigen::Matrix3d{};

    const Eigen::Matrix3d K_inv = K.inverse();

    SARA_DEBUG << "Filling points and rays for P3P..." << std::endl;
    auto xs = std::array{0, 1, 0};
    auto ys = std::array{0, 0, 1};
    for (auto n = 0; n < 3; ++n)
    {
      const auto& x = xs[n];
      const auto& y = ys[n];
      const Eigen::Vector3d xn =
          cb.image_point(x, y).homogeneous().cast<double>();
      if (is_nan(xn))
        continue;

      points.col(n) = cb.scene_point(x, y);
      rays.col(n) = (K_inv * xn).normalized();
    }
    if (is_nan(points) || is_nan(rays))
      return std::nullopt;

    SARA_DEBUG << "Solving P3P..." << std::endl;
    SARA_DEBUG << "Points =\n" << points << std::endl;
    SARA_DEBUG << "Rays   =\n" << rays << std::endl;
    const auto poses = solve_p3p(points, rays);
    if (poses.empty())
      return std::nullopt;

    // Find the best poses.
    SARA_DEBUG << "Determining the best pose..." << std::endl;
    auto errors = std::vector<double>{};
    for (const auto& pose : poses)
    {
      auto error = 0;

      auto n = 0;
      for (auto y = 0; y < cb.height(); ++y)
      {
        for (auto x = 0; x < cb.width(); ++x)
        {
          auto x0 = cb.image_point(x, y);
          if (is_nan(x0))
            continue;

          const auto& R = pose.topLeftCorner<3, 3>();
          const auto& t = pose.col(3);

          const Eigen::Vector2f x1 =
              (K * (R * cb.scene_point(x, y) + t)).hnormalized().cast<float>();
          error += (x1 - x0).squaredNorm();
          ++n;
        }
      }

      errors.emplace_back(error / n);
    }

    const auto best_pose_index =
        std::min_element(errors.begin(), errors.end()) - errors.begin();
    const auto& best_pose = poses[best_pose_index];
    SARA_DEBUG << "Best pose:\n" << best_pose << std::endl;

    return best_pose;
  }


  template <typename CameraModel>
  inline auto inspect(ImageView<Rgb8>& image,               //
                      const ChessboardCorners& chessboard,  //
                      const CameraModel& camera,            //
                      const Eigen::Matrix3d& R,             //
                      const Eigen::Vector3d& t,             //
                      bool pause = false) -> void
  {
    const auto s = chessboard.square_size().value;

    const Eigen::Vector3d& o3 = t;
    const Eigen::Vector3d i3 = R * Eigen::Vector3d::UnitX() * s + t;
    const Eigen::Vector3d j3 = R * Eigen::Vector3d::UnitY() * s + t;
    const Eigen::Vector3d k3 = R * Eigen::Vector3d::UnitZ() * s + t;
    const Eigen::Vector2f o = camera.project(o3).template cast<float>();
    const Eigen::Vector2f i = camera.project(i3).template cast<float>();
    const Eigen::Vector2f j = camera.project(j3).template cast<float>();
    const Eigen::Vector2f k = camera.project(k3).template cast<float>();

    static const auto red = Rgb8{167, 0, 0};
    static const auto green = Rgb8{89, 216, 26};
    draw_arrow(image, o, i, red, 6);
    draw_arrow(image, o, j, green, 6);
    draw_arrow(image, o, k, Blue8, 6);

    for (auto y = 0; y < chessboard.height(); ++y)
    {
      for (auto x = 0; x < chessboard.width(); ++x)
      {
        Eigen::Vector3d P = chessboard.scene_point(x, y).cast<double>();
        P = R * P + t;

        const Eigen::Vector2f p1 = chessboard.image_point(x, y);
        const Eigen::Vector2f p2 = camera.project(P).template cast<float>();

        if (!is_nan(p1))
          draw_circle(image, p1, 3.f, Cyan8, 3);
        draw_circle(image, p2, 3.f, Magenta8, 3);
        if (pause)
        {
          display(image);
          get_key();
        }
      }
    }
  }

}  // namespace DO::Sara
