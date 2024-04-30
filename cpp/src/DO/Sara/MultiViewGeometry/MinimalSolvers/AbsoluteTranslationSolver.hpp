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

#include <Eigen/Dense>
#include <Eigen/Eigen>


namespace DO::Sara {

  //! @brief Two-point solver that solves the absolute translation given a known
  //!        absolute rotaton.
  //!
  //! Turns out to be very basic...
  template <typename T>
  struct AbsoluteTranslationSolver
  {
    static constexpr auto num_points = 2;
    static constexpr auto num_models = 1;

    using model_type = Eigen::Vector3<T>;

    //! @brief Inputs are rotated world scene points and the backprojected rays.
    auto operator()(const Eigen::Matrix<T, 3, 2>& Rx,
                    const Eigen::Matrix<T, 3, 2>& y) const -> model_type
    {
      const auto x0 = Rx.col(0);
      const auto x1 = Rx.col(1);
      const auto y0 = y.col(0);
      const auto y1 = y.col(1);

      static const auto I3 = Eigen::Matrix3<T>::Identity();
      static const auto O3 = Eigen::Vector3<T>::Zero();

      auto A = Eigen::Matrix<T, 6, 5>{};
      A.template topRows<3>() << I3, y0, O3;
      A.template bottomRows<3>() << I3, O3, y1;

      auto b = Eigen::Vector<T, 6>{};
      b << x0, x1;

      return A.template colPivHouseholderQr().solve(b);
    }
  };

}  // namespace DO::Sara
#pragma once
