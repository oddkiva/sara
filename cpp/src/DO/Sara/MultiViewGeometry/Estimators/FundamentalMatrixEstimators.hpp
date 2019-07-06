// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>

#include <array>


namespace DO::Sara {

  struct DO_SARA_EXPORT EightPointAlgorithm
  {
    using model_type = FundamentalMatrix;
    using matrix_type = Eigen::Matrix<double, 3, 8>;
    using matrix_view_type = Eigen::Map<const matrix_type>;

    static constexpr auto num_points = 8;

    auto operator()(const matrix_view_type& x, const matrix_view_type& y) const
        -> std::array<model_type, 1>;

    auto operator()(const matrix_type& x, const matrix_type& y) const
        -> std::array<model_type, 1>
    {
      const auto x_view = matrix_view_type{x.data()};
      const auto y_view = matrix_view_type{y.data()};
      return this->operator()(x_view, y_view);
    }
  };

} /* namespace DO::Sara */
