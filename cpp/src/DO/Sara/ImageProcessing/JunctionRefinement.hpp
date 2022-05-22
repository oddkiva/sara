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

#pragma once

#include <Eigen/Eigen>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/Math/UsualFunctions.hpp>


namespace DO::Sara {

  // Forstner corner (actually junction) refinement method
  //
  // min_c Sum_x <g_x, (x - c)>^2
  //
  // Sum_x <g_x, x - c> [1, 1]^T= [0, 0]^T
  // Sum_x <g_x, x> = Sum_x <g_x, c>
  // cx = Sum_x (g_x[0] * x[0]) / Sum_x g_x[0]
  inline auto
  refine_junction_location_unsafe(const ImageView<Eigen::Vector2f>& grad,
                                  const Eigen::Vector2i& x0,  //
                                  const std::int32_t r) -> Eigen::Vector2f
  {
    Eigen::Matrix2f A = Eigen::Matrix2f::Zero();
    Eigen::Vector2f b = Eigen::Vector2f::Zero();
    for (auto dy = -r; dy <= r; ++dy)
    {
      for (auto dx = -r; dx <= r; ++dx)
      {
        const Eigen::Vector2i x1 = x0 + Eigen::Vector2i(dx, dy);
        const auto& g = grad(x1);

        // Update A.
        A (0, 0) += g.x() * g.x(); A(0, 1) += g.x() * g.y();
        A (1, 0) += g.x() * g.y(); A (1, 1) += g.y() * g.y();

        // Update b.
        b += g * g.dot(x1.cast<float>());
      }
    }

    return (A.transpose() * A).ldlt().solve(A.transpose() * b);
  }

}  // namespace DO::Sara
