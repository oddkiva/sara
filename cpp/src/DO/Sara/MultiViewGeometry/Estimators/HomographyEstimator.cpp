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

#include <DO/Sara/MultiViewGeometry/Estimators/HomographyEstimator.hpp>


namespace DO { namespace Sara {

  auto FourPointAlgorithm::
  operator()(const FourPointAlgorithm::matrix_view_type& x,
             const FourPointAlgorithm::matrix_view_type& y) const
      -> std::array<FourPointAlgorithm::model_type, 1>
  {
    const auto zero = RowVector3d::Zero();

    auto M = Matrix<double, 8, 8>{};
    for (int i = 0; i < 4; ++i)
    {
      RowVector3d u_i = x.col(i).transpose();
      RowVector3d v_i = y.col(i).transpose();
      M.row(2* i + 0) <<  u_i, zero, - u_i.head(2) * v_i.x();
      M.row(2* i + 1) << zero,  u_i, - u_i.head(2) * v_i.y();
    }

    const Matrix<double, 2, 4> y_euclidean = y.topRows<2>();
    const Map<const Matrix<double, 8, 1>> b{y_euclidean.data()};

    const Matrix<double, 8, 1> h = M.colPivHouseholderQr().solve(b);

    auto H = Homography{};

    H.matrix().row(0) = h.segment(0, 3).transpose();
    H.matrix().row(1) = h.segment(3, 3).transpose();
    H.matrix().row(2) << h.segment(6, 2).transpose(), 1;

    return {H};
  }

} /* namespace Sara */
} /* namespace DO */
