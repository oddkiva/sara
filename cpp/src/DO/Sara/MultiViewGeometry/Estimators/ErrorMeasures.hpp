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
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>


namespace DO::Sara {

  //! @brief Functor evaluating distance of a point to its epipolar line.
  struct EpipolarDistance
  {
    EpipolarDistance() = default;

    EpipolarDistance(const Eigen::Matrix3d& F_)
      : F{F_}
    {
    }

    auto operator()(const Eigen::Vector3d& X, const Eigen::Vector3d& Y) const
    {
      return std::abs(Y.transpose() * F * X);
    }

    template <typename Mat>
    auto operator()(const Mat& X, const Mat& Y) const -> RowVectorXd
    {
      return (Y.array() * (F * X).array()).colwise().sum().abs();
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

    template <typename Mat>
    auto operator()(const Mat& u1, const Mat& u2) const -> RowVectorXd
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


  //! @brief Joint cheirality and epipolar consistency for RANSAC.
  struct CheiralAndEpipolarConsistency
  {
    using Model = TwoViewGeometry;

    Model geometry;
    EpipolarDistance distance;
    double err_threshold;

    auto set_model(const Model& g)
    {
      geometry = g;
    }

    // N.B.: this is not a const method. This triangulates the points from the
    // point correspondences and updates the cheirality.
    template <typename Mat>
    auto operator()(const Mat& u1, const Mat& u2)
        -> Eigen::Array<bool, 1, Eigen::Dynamic>
    {
      const auto epipolar_consistent = distance(u1, u2).array() < err_threshold;

      const Matrix34d P1 = geometry.C1;
      const Matrix34d P2 = geometry.C2;

      auto& X = geometry.X;
      auto& cheirality = geometry.cheirality;

      X = triangulate_linear_eigen(P1, P2, u1, u2);
      cheirality = relative_motion_cheirality_predicate(X, P2);

      return epipolar_consistent && cheirality;
    }
  };

} /* namespace DO::Sara */
