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
#include <DO/Sara/Core/Math/PolynomialRoots.hpp>


namespace DO::Sara {

  auto fit_ellipse(const std::vector<Eigen::Vector2f>& points)
      -> Eigen::Matrix3f;


  inline auto y_from_conic(const Eigen::Matrix3f& E, const float x)
  {
    auto p = UnivariatePolynomial<float, 2>{};
    const auto& a = E(0, 0);
    const auto& b = E(1, 1);
    const auto& c = E(0, 1);
    const auto& d = E(0, 2);
    const auto& e = E(1, 2);
    const auto& f = E(2, 2);

    p[0] = a * x * x + 2 * d * x + f;
    p[1] = 2 * (c * x + e);
    p[2] = b;

    float y1, y2;
    const auto found = compute_quadratic_real_roots(p, y1, y2);
    if (!found)
      std::runtime_error{"Something wrong"};

    return std::make_pair(y1, y2);
  }

  inline auto x_from_conic(const Eigen::Matrix3f& E, const float y)
  {
    auto p = UnivariatePolynomial<float, 2>{};
    const auto& a = E(0, 0);
    const auto& b = E(1, 1);
    const auto& c = E(0, 1);
    const auto& d = E(0, 2);
    const auto& e = E(1, 2);
    const auto& f = E(2, 2);

    p[0] = b * y * y + 2 * e * y + f;
    p[1] = 2 * (c * y + d);
    p[2] = a;

    float x1, x2;
    const auto found = compute_quadratic_real_roots(p, x1, x2);
    if (!found)
      std::runtime_error{"Something wrong"};

    return std::make_pair(x1, x2);
  }

}  // namespace DO::Sara
