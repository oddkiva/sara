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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Math/PolynomialRoots.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>


namespace DO::Sara {

  template <typename T>
  struct SevenPointAlgorithmImpl
  {
    static constexpr auto num_points = 7;
    static constexpr auto num_models = 3;

    using data_point_type = Eigen::Matrix<T, 4, 7>;

    static auto extract_nullspace(const data_point_type& X)
        -> std::array<Eigen::Matrix3<T>, 2>
    {
      // 1. solve the linear system from the 8-point correspondences.
      auto A = Eigen::Matrix<T, 7, 9>{};
      for (int i = 0; i < X.cols(); ++i)
      {
        const auto p_left = X.col(i).head(2).homogeneous();
        const auto p_right = X.col(i).tail(2);
        // clang-format off
        A.row(i) << p_right(0) * p_left.transpose(),
                    p_right(1) * p_left.transpose(),
                    p_left.transpose();
        // clang-format on
      }

      auto svd = Eigen::BDCSVD<Matrix<T, 7, 9>>{A, Eigen::ComputeFullV};
      const Eigen::Vector<T, 9> f1 = svd.matrixV().col(7);
      const Eigen::Vector<T, 9> f2 = svd.matrixV().col(8);

      const auto to_matrix = [](const auto& f) {
        auto F = Eigen::Matrix3<T>{};
        F.row(0) = f.segment(0, 3).transpose();
        F.row(1) = f.segment(3, 3).transpose();
        F.row(2) = f.segment(6, 3).transpose();
        return F;
      };

      return {to_matrix(f1), to_matrix(f2)};
    }

    static auto form_determinant_constraint(const Eigen::Matrix3<T>& F1,
                                            const Eigen::Matrix3<T>& F2)
    {
      auto P = UnivariatePolynomial<T, 3>{};

      // Lambda-Twist has a nice formula for the determinant. Let's reuse it
      // instead of using SymPy.
      //
      // clang-format off
      P[3] = F2.determinant();

      P[2] = F1.col(0).dot(F2.col(1).cross(F2.col(2))) +
             F1.col(1).dot(F2.col(2).cross(F2.col(0))) +
             F1.col(2).dot(F2.col(0).cross(F2.col(1)));

      P[1] = F2.col(0).dot(F1.col(1).cross(F1.col(2))) +
             F2.col(1).dot(F1.col(2).cross(F1.col(0))) +
             F2.col(2).dot(F1.col(0).cross(F1.col(1)));

      P[0] = F1.determinant();
      // clang-format on

      return P;
    }

    static auto
    solve_determinant_constraint(const std::array<Eigen::Matrix3<T>, 2>& F)
        -> std::vector<Eigen::Matrix3<T>>
    {
      // Because the fundamental matrix is rank 2, the determinant must be 0,
      // i.e.: det(F[0] + α F[1]) = 0
      // This is a cubic polynomial in α.
      const auto det_F = form_determinant_constraint(F[0], F[1]);

      // We determine 3 real roots α_i at most.
      auto α = std::array<T, num_models>{};
      const auto all_real_roots = compute_cubic_real_roots(det_F,  //
                                                           α[0], α[1], α[2]);
      const auto num_real_roots = all_real_roots ? 3 : 1;

      // Form the candidate fundamental matrices.
      auto F0 = std::vector<Eigen::Matrix3<T>>(num_real_roots);
      std::transform(α.begin(), α.begin() + num_real_roots, F0.begin(),
                     [&F](const auto& α_i) -> Eigen::Matrix3<T> {
                       // Normalize the fundamental matrix to avoid numerical
                       // issues and later headaches.
                       return (F[0] + α_i * F[1]).normalized();
                     });

      return F0;
    }
  };

  struct DO_SARA_EXPORT SevenPointAlgorithmDoublePrecision
  {
    using Impl = SevenPointAlgorithmImpl<double>;

    static constexpr auto num_points = Impl::num_points;
    static constexpr auto num_models = Impl::num_models;

    using internal_data_point_type = Impl::data_point_type;
    using data_point_type = std::array<TensorView_<double, 2>, 2>;
    using model_type = FundamentalMatrix;

    static auto extract_nullspace(const internal_data_point_type& X)
        -> std::array<Eigen::Matrix3d, 2>;

    static auto form_determinant_constraint(const Eigen::Matrix3d& F1,
                                            const Eigen::Matrix3d& F2)
        -> UnivariatePolynomial<double, 3>;

    static auto
    solve_determinant_constraint(const std::array<Eigen::Matrix3d, 2>& F)
        -> std::vector<Eigen::Matrix3d>;

    auto operator()(const internal_data_point_type& X) const
        -> std::vector<Eigen::Matrix3d>;

    auto operator()(const data_point_type& X) const
        -> std::vector<Eigen::Matrix3d>
    {
      auto Xi = internal_data_point_type{};
      const Eigen::Matrix<double, 2, 7> X0 =
          X[0].colmajor_view().matrix().colwise().hnormalized();
      const Eigen::Matrix<double, 2, 7> X1 =
          X[1].colmajor_view().matrix().colwise().hnormalized();
      Xi << X0, X1;
      return this->operator()(Xi);
    }

    Impl _impl;
  };


  template <typename EpipolarDistance>
  inline auto
  normalized_epipolar_residual(const std::vector<std::size_t>& subset,
                               const Eigen::Matrix3d& F,
                               const std::vector<Eigen::Vector4d>& matches,
                               const EpipolarDistance& distance)
      -> std::vector<double>
  {
    return normalized_residual(subset, F, matches, distance);
  }

}  // namespace DO::Sara
