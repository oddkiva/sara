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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO::Sara {

  //! @ingroup MultiViewGeometry
  //! @defgroup MultiviewErrorMeasures Error Measures
  //! @{

  //! @brief Functor evaluating distance of a point to its epipolar line.
  struct EpipolarDistance
  {
    auto operator()(const Eigen::Vector3d& X, const Eigen::Vector3d& Y) const
    {
      return std::abs(Y.transpose() * F * X);
    }

    template <typename Derived>
    auto operator()(const Eigen::MatrixBase<Derived>& X,
                    const Eigen::MatrixBase<Derived>& Y) const
        -> Eigen::RowVectorXd
    {
      return (Y.array() * (F * X).array()).colwise().sum().abs();
    }

    Eigen::Matrix3d F;
  };

  //! See for example the following reference:
  //! https://cseweb.ucsd.edu/classes/sp04/cse252b/notes/lec11/lec11.pdf
  struct SampsonEpipolarDistance
  {
    auto operator()(const Eigen::Vector3d& X, const Eigen::Vector3d& Y) const
    {
      const Eigen::Vector3d Xn = X / X.z();
      const Eigen::Vector3d Yn = Y / Y.z();
      const double sqrt_num = Yn.transpose() * F * Xn;

      const auto right_line = F * Xn;
      const auto left_line = F.transpose() * Yn;
      const double den = right_line.head(2).squaredNorm() +  //
                         left_line.head(2).squaredNorm();

      return (sqrt_num * sqrt_num) / den;
    }

    template <typename Derived>
    auto operator()(const Eigen::MatrixBase<Derived>& X,
                    const Eigen::MatrixBase<Derived>& Y) const
        -> Eigen::RowVectorXd
    {
      const auto Xn = X.colwise().hnormalized().colwise().homogeneous();
      const auto Yn = Y.colwise().hnormalized().colwise().homogeneous();
      const auto dots = (Yn.array() * (F * Xn).array())
                           .matrix()
                           .colwise()
                           .sum();
      const auto nums = dots.array().square();
      // SARA_CHECK(nums);

      const auto right_lines = F * Xn;
      const auto left_lines = F.transpose() * Yn;
      const auto dens =
          (right_lines.template topRows<2>().colwise().squaredNorm() +  //
           left_lines.template topRows<2>().colwise().squaredNorm())
              .array();
      // SARA_CHECK(dens);

      const auto residuals = (nums / dens).matrix();
      // SARA_CHECK(residuals);

      return residuals;
    }

    Eigen::Matrix3d F;
  };

  struct SymmetricTransferError
  {
    SymmetricTransferError() = default;

    SymmetricTransferError(const Eigen::Matrix3d& H)
      : H_{H}
      , H_inv_{H.inverse()}
    {
    }

    auto operator()(const Eigen::Vector3d& x, const Eigen::Vector3d& y) const
        -> double
    {
      return ((H_ * x).hnormalized() - y.hnormalized()).norm() +
             ((H_inv_ * y).hnormalized() - x.hnormalized()).norm();
    }

    template <typename Derived>
    auto operator()(const Eigen::MatrixBase<Derived>& u1,
                    const Eigen::MatrixBase<Derived>& u2) const
        -> Eigen::RowVectorXd
    {
      auto hu1_u2 =
          (H_ * u1).colwise().hnormalized() - u2.colwise().hnormalized();
      auto hinv_u2_u1 =
          (H_inv_ * u2).colwise().hnormalized() - u1.colwise().hnormalized();

      auto hu1_u2_norm = hu1_u2.colwise().norm();
      auto hinv_u2_u1_norm = hinv_u2_u1.colwise().norm();
      return hu1_u2_norm + hinv_u2_u1_norm;
    }

    Eigen::Matrix3d H_;
    Eigen::Matrix3d H_inv_;
  };

  //! @}

} /* namespace DO::Sara */
