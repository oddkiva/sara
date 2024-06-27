// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/MultiViewGeometry/MinimalSolvers/SteweniusFivePointAlgorithm.hpp>


using namespace DO::Sara;


auto SteweniusFivePointAlgorithm::extract_null_space(
    const Eigen::Matrix<double, 3, 5>& p_left,
    const Eigen::Matrix<double, 3, 5>& p_right) const
    -> Eigen::Matrix<double, 9, 4>
{
  auto A = Eigen::Matrix<double, 5, 9>{};

  for (int i = 0; i < 5; ++i)
    A.row(i) <<                                     //
        p_right(0, i) * p_left.col(i).transpose(),  //
        p_right(1, i) * p_left.col(i).transpose(),  //
        p_right(2, i) * p_left.col(i).transpose();

  // The essential matrix lives in right null space of A.
  const Eigen::Matrix<double, 9, 4> K =
      A.bdcSvd(Eigen::ComputeFullV).matrixV().rightCols(4);
  return K;
}

auto SteweniusFivePointAlgorithm::build_essential_matrix_constraints(
    const Eigen::Matrix<double, 9, 4>& E_bases) const
    -> Eigen::Matrix<double, 10, 20>
{
  const auto X = E_bases.col(0).data();
  const auto Y = E_bases.col(1).data();
  const auto Z = E_bases.col(2).data();
  const auto W = E_bases.col(3).data();

  auto A = Eigen::Matrix<double, 10, 20>{};
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/Stewenius/EssentialMatrixPolynomialConstraints.hpp>
  return A;
}

auto SteweniusFivePointAlgorithm::solve_essential_matrix_constraints(
    const Eigen::Matrix<double, 9, 4>& E_bases,
    const Eigen::Matrix<double, 10, 20>& M) const
    -> std::vector<EssentialMatrix>
{
  // This follows the Matlab code at the end of section 4. of "Recent
  // Developments on Direct Relative Orientation", Stewenius et al.

  using Matrix10d = Eigen::Matrix<double, 10, 10>;

  auto lu = Eigen::FullPivLU<Matrix10d>{M.block<10, 10>(0, 0)};
  const Matrix10d B = lu.solve(M.block<10, 10>(0, 10));

  Matrix10d At = Matrix10d::Zero();

  // The following does:
  // At.row(0) = -B.row(0);
  // At.row(1) = -B.row(1);
  // At.row(2) = -B.row(2);
  // At.row(3) = -B.row(4);
  // At.row(4) = -B.row(5);
  // At.row(5) = -B.row(7);
  At.block<3, 10>(0, 0) = -B.block<3, 10>(0, 0);
  At.block<2, 10>(3, 0) = -B.block<2, 10>(4, 0);
  At.row(5) = -B.row(7);

  At(6, 0) = 1;
  At(7, 1) = 1;
  At(8, 3) = 1;
  At(9, 6) = 1;

  Eigen::EigenSolver<Matrix10d> eigs(At);
  const MatrixXcd U = eigs.eigenvectors();
  const VectorXcd V = eigs.eigenvalues();

  // Build essential matrices for the real solutions.
  auto Es = std::vector<EssentialMatrix>{};
  Es.reserve(10);

  for (int s = 0; s < 10; ++s)
  {
    // Only consider real solutions.
    if (V(s).imag() != 0)
      continue;

    // N.B.: the span of the nullspace E_bases are the set of flattened matrices
    // where originally the coefficients are stored in row-major order.
    auto E = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>{};
    auto vec_E = Eigen::Map<Eigen::Vector<double, 9>>{E.data()};
    vec_E = E_bases * U.col(s).tail<4>().real();

    // Normalizing the essential matrix will make sure the epipolar line-point
    // distances have nice value and it's useful to do it before counting the
    // inliers in RANSAC.
    E.normalize();

    Es.emplace_back(E);
  }

  return Es;
}

auto SteweniusFivePointAlgorithm::operator()(
    const Matrix<double, 3, 5>& left,  //
    const Matrix<double, 3, 5>& right) const -> std::vector<EssentialMatrix>
{
  // 1. Extract the null space.
  const auto E_bases = extract_null_space(left, right);

  // 2. Build the polynomial system that the essential matrix must satisfy.
  //    This is fast because each coefficient of the polynomial system is
  //    precomputed with SymPy.
  const auto E_constraints = build_essential_matrix_constraints(E_bases);
  const auto Es = solve_essential_matrix_constraints(E_bases, E_constraints);
  return Es;
}
