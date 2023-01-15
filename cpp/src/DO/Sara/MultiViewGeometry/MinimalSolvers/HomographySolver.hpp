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

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/Homography.hpp>

#include <array>


namespace DO::Sara {

  //! @addtogroup MinimalSolvers
  //! @{

  //! @brief Four-point algorithm for the homography.
  struct DO_SARA_EXPORT FourPointAlgorithm
  {
    using model_type = Homography;
    using matrix_type = Eigen::Matrix<double, 3, 4>;
    using matrix_view_type = Eigen::Map<const matrix_type>;
    using data_point_type = std::array<TensorView_<double, 2>, 2>;

    static constexpr auto num_points = 4;
    static constexpr auto num_models = 1;

    //! N.B.: the 2D data points are required in homogeneous coordinates (thus
    //! 3D).
    auto operator()(const matrix_view_type& x, const matrix_view_type& y) const
        -> std::array<model_type, num_models>;

    //! N.B.: the 2D data points are required in homogeneous coordinates (thus
    //! 3D).
    auto operator()(const matrix_type& x, const matrix_type& y) const
        -> std::array<model_type, num_models>
    {
      const auto x_view = matrix_view_type{x.data()};
      const auto y_view = matrix_view_type{y.data()};
      return this->operator()(x_view, y_view);
    }

    auto operator()(const data_point_type& X) const
        -> std::array<model_type, num_models>
    {
      return this->operator()(X[0].colmajor_view().matrix(),
                              X[1].colmajor_view().matrix());
    }
  };

  //! @}

}  // namespace DO::Sara
