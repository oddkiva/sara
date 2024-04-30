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

#include <DO/Sara/MultiViewGeometry/MinimalSolvers/SevenPointAlgorithm.hpp>


namespace DO::Sara {

  auto SevenPointAlgorithmDoublePrecision::extract_nullspace(
      const Eigen::Matrix<double, 4, 7>& X) -> std::array<Eigen::Matrix3d, 2>
  {
    return Impl::extract_nullspace(X);
  }

  auto SevenPointAlgorithmDoublePrecision::form_determinant_constraint(
      const Eigen::Matrix3d& F1, const Eigen::Matrix3d& F2)
      -> UnivariatePolynomial<double, 3>
  {
    return Impl::form_determinant_constraint(F1, F2);
  }

  auto SevenPointAlgorithmDoublePrecision::solve_determinant_constraint(
      const std::array<Eigen::Matrix3d, 2>& F) -> std::vector<Eigen::Matrix3d>
  {
    return Impl::solve_determinant_constraint(F);
  }

  auto SevenPointAlgorithmDoublePrecision::operator()(
      const Eigen::Matrix<double, 4, 7>& X) const
      -> std::vector<Eigen::Matrix3d>
  {
    // The fundamental matrix lives in the nullspace of data matrix X, which
    // has rank 2, i.e., Null(X) = Span(F[0], F[1])
    //
    // The fundamental matrix is a linear combination F[0] + α F[1].
    const auto F = extract_nullspace(X);

    // Extract the roots of polynomial in α: det(F[0] + α F[1]).
    //
    // We get three possible solutions at most.
    return solve_determinant_constraint(F);
  }

}  // namespace DO::Sara
