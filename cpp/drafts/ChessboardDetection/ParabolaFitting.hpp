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

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


auto normalization_transform(const std::vector<Eigen::Vector2f>& points)
    -> Eigen::Matrix3f
{
  auto T = Eigen::Matrix3f{};
  T.setIdentity();
  const auto [minxi, maxxi] = std::minmax_element(
      points.begin(), points.end(),
      [](const auto& a, const auto& b) { return a.x() < b.x(); });
  const auto [minyi, maxyi] = std::minmax_element(
      points.begin(), points.end(),
      [](const auto& a, const auto& b) { return a.y() < b.y(); });

  const auto minx = minxi->x();
  const auto maxx = maxxi->x();
  const auto miny = minyi->y();
  const auto maxy = maxyi->y();

  const auto diffx = maxx - minx;
  const auto diffy = maxy - miny;

  T(0, 0) = 1 / diffx;
  T(1, 1) = 1 / diffy;
  T(0, 2) = -minx / diffx;
  T(1, 2) = -miny / diffy;

  return T;
}

auto apply(const Eigen::Matrix3f& T, const std::vector<Eigen::Vector2f>& points)
    -> std::vector<Eigen::Vector2f>
{
  auto points_normalized = points;
  for (auto& p : points_normalized)
    p = (T * p.homogeneous()).hnormalized();
  return points_normalized;
}


auto y_parabola(const std::vector<Eigen::Vector2f>& points) -> Eigen::Vector3f
{
  auto A = Eigen::MatrixXf(points.size(), 3);
  auto b = Eigen::VectorXf(points.size());
  for (auto i = 0u; i < points.size(); ++i)
  {
    const auto& x = points[i].x();
    const auto& y = points[i].y();
    A.row(i) << x * x, x, 1;
    b(i) = y;
  }
  const Eigen::Vector3f f = A.fullPivHouseholderQr().solve(b);
  return f;
}

auto x_parabola(const std::vector<Eigen::Vector2f>& points) -> Eigen::Vector3f
{
  auto A = Eigen::MatrixXf(points.size(), 3);
  auto b = Eigen::VectorXf(points.size());
  for (auto i = 0u; i < points.size(); ++i)
  {
    const auto& x = points[i].x();
    const auto& y = points[i].y();
    A.row(i) << y * y, y, 1;
    b(i) = x;
  }
  const Eigen::Vector3f f = A.fullPivHouseholderQr().solve(b);
  return f;
}
