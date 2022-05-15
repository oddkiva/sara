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

  // Forstner corner refinement method
  //
  // min_c Sum_x <g_x, (x - c)>^2
  //
  // Sum_x <g_x, x - c> [1, 1]^T= [0, 0]^T
  // Sum_x <g_x, x> = Sum_x <g_x, c>
  // cx = Sum_x (g_x[0] * x[0]) / Sum_x g_x[0]
  inline auto
  refine_corner_location_unsafe(const ImageView<Eigen::Vector2f>& grad,
                                const Eigen::Vector2i& x0,  //
                                const std::int32_t r) -> Eigen::Vector2f
  {
    Eigen::Vector2f x = Eigen::Vector2f::Zero();
    Eigen::Vector2f sum = Eigen::Vector2f::Zero();
    for (auto dy = -r; dy <= r; ++dy)
    {
      for (auto dx = -r; dx <= r; ++dx)
      {
        const Eigen::Vector2i x1 = x0 + Eigen::Vector2i(dx, dy);
        const auto& g = grad(x1);
        x.x() += g.x() * x1.x();
        x.y() += g.y() * x1.y();
        sum += g;
      }
    }

    x.array() /= sum.array();

    return x;
  }

}  // namespace DO::Sara
