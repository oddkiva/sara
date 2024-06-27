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

#include <DO/Sara/MultiViewGeometry/MinimalSolvers/NisterFivePointAlgorithm.hpp>

#include <DO/Sara/Core/Math/JenkinsTraub.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>


using namespace DO::Sara;


auto NisterFivePointAlgorithm::extract_null_space(
    const Eigen::Matrix<double, 3, 5>& p_left,
    const Eigen::Matrix<double, 3, 5>& p_right) const -> Matrix<double, 9, 4>
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

auto NisterFivePointAlgorithm::build_essential_matrix_constraints(
    const double X[9],  //
    const double Y[9],  //
    const double Z[9],  //
    const double W[9]) const -> Eigen::Matrix<double, 10, 20>
{
  auto A = Eigen::Matrix<double, 10, 20>{};
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/Nister/EssentialMatrixPolynomialConstraints.hpp>
  return A;
}

auto NisterFivePointAlgorithm::inplace_gauss_jordan_elimination(
    Eigen::Matrix<double, 10, 20>& U) const -> void
{
#if 0
  // We should use Eigen LU algorithm as it is more stable numerically.
  U = A.fullPivLu().matrixLU().triangularView<Eigen::Upper>();

  // Rescale the first monomial coefficient to coefficient 1.
  for (auto i = 0; i < 10; ++i)
    U.row(i) *= 1 / U(i, i);

  // Simplify the upper triangular matrix further.
  for (auto i = 9; i >= 4; --i)
  {
    for (auto j = i - 1; j >= 4; --j)
    {
      U.row(j) = U.row(j) / U(j, i) - U.row(i);
      U.row(j) /= U(j, j);
    }
  }
#else
  for (auto i = 0; i < 10; ++i)
  {
    U.row(i) /= U(i, i);
    for (auto j = i + 1; j < 10; ++j)
      U.row(j) = U.row(j) / U(j, i) - U.row(i);
  }

  for (auto i = 0; i < 10; ++i)
    U.row(i) *= 1 / U(i, i);

  for (auto i = 9; i >= 4; --i)
  {
    for (auto j = i - 1; j >= 4; --j)
    {
      U.row(j) = U.row(j) / U(j, i) - U.row(i);
      U.row(j) /= U(j, j);
    }
  }
#endif
}

auto NisterFivePointAlgorithm::calculate_resultant_determinant(
    const double S[6 * 10]) const -> UnivariatePolynomial<double>
{
  auto n = UnivariatePolynomial<double>{10};
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/Nister/EssentialMatrixResultingDeterminant.hpp>
  return n;
}

auto NisterFivePointAlgorithm::calculate_resultant_determinant_minors(
    const double S[6 * 10]) const -> std::array<UnivariatePolynomial<double>, 3>
{
  auto p = std::array{UnivariatePolynomial<double>{7},
                      UnivariatePolynomial<double>{7},
                      UnivariatePolynomial<double>{6}};
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/Nister/EssentialMatrixResultingMinor_0.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/Nister/EssentialMatrixResultingMinor_1.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/Nister/EssentialMatrixResultingMinor_2.hpp>
  return p;
}

auto NisterFivePointAlgorithm::solve_reduced_constraint_system(
    const Eigen::Matrix<double, 6, 10, Eigen::RowMajor>& U_reduced,
    const double X[9],  //
    const double Y[9],  //
    const double Z[9],  //
    const double W[9]) const -> std::vector<EssentialMatrix>
{
  const auto S = U_reduced.data();

  const auto n = calculate_resultant_determinant(S);
  const auto [roots_extracted_successfully, roots] = rpoly(n);
  // If rpoly fails, then let's take a conservative behaviour, minimize the
  // risk of providing false essential matrices...
  //
  // It probably means the polynomial have poorly conditioned coefficients, and
  // that the point correspondences were wrong anyways.
  if (!roots_extracted_successfully)
    return {};

  const auto p = calculate_resultant_determinant_minors(S);

  auto xyzs = std::vector<Vector3d>{};
  xyzs.reserve(10);
  for (const auto& z_complex : roots)
  {
    if (z_complex.imag() != 0)
      continue;

    const auto z = z_complex.real();

    const auto p0_z = p[0](z);
    const auto p1_z = p[1](z);
    const auto p2_z = p[2](z);

    const auto x = p0_z / p2_z;
    const auto y = p1_z / p2_z;

    if (std::isnan(x) || std::isinf(x) || std::isnan(y) || std::isinf(y))
      continue;

    xyzs.emplace_back(x, y, z);
  }

  // 4. Build essential matrices for the real solutions.
  auto Es = std::vector<EssentialMatrix>{};
  Es.reserve(10);

  using ConstMatView3d =
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>;
  const auto X33 = ConstMatView3d{X};
  const auto Y33 = ConstMatView3d{Y};
  const auto Z33 = ConstMatView3d{Z};
  const auto W33 = ConstMatView3d{W};
  for (const auto& xyz : xyzs)
  {
    const auto& x = xyz[0];
    const auto& y = xyz[1];
    const auto& z = xyz[2];

    auto E = EssentialMatrix{};
    E.matrix() = x * X33 + y * Y33 + z * Z33 + W33;

    // Normalizing the essential matrix will make sure the epipolar line-point
    // distances have nice value and it's useful to do it before counting the
    // inliers in RANSAC.
    E.matrix().normalize();

    Es.emplace_back(E);
  }

  return Es;
}

auto NisterFivePointAlgorithm::find_essential_matrices(
    const Eigen::Matrix<double, 3, 5>& x1,
    const Eigen::Matrix<double, 3, 5>& x2) const -> std::vector<EssentialMatrix>
{
  // 1. Extract the null space.
  const auto E_bases = extract_null_space(x1, x2);

  // 2. Build the polynomial system that the essential matrix must satisfy.
  //    This is fast because each coefficient of the polynomial system is
  //    precomputed with SymPy.
  const auto X = E_bases.col(0).data();
  const auto Y = E_bases.col(1).data();
  const auto Z = E_bases.col(2).data();
  const auto W = E_bases.col(3).data();
  auto A = build_essential_matrix_constraints(X, Y, Z, W);

  // 3. Gauss-Jordan elimination.
  inplace_gauss_jordan_elimination(A);

  // 4. Extract the resultant matrix from the upper triangular matrix.
  //    Using some clever algebraic operation, we find that necessarily
  //    [x, y, 1]^T must live in the nullspace of some 3x3 matrix `B`.
  //
  //    The coefficients of matrix B are polynomials in the variable `z`.
  //    So that means the polynomial det(B)(z) = 0.
  const Eigen::Matrix<double, 6, 10, Eigen::RowMajor> A_reduced =
      A.bottomRightCorner<6, 10>();
  const auto Es = solve_reduced_constraint_system(A_reduced, X, Y, Z, W);
  return Es;
}
