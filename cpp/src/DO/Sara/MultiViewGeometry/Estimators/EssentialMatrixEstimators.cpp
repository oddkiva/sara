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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Math/JenkinsTraub.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/EssentialMatrixEstimators.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>

//#define SHOW_DEBUG_LOG


using namespace std;


namespace DO::Sara {

auto FivePointAlgorithmBase::extract_null_space(
    const Matrix<double, 3, 5>& p_left,
    const Matrix<double, 3, 5>& p_right) const -> Matrix<double, 9, 4>
{
  auto A = Matrix<double, 5, 9>{};

  for (int i = 0; i < 5; ++i)
    A.row(i) <<                                     //
        p_right(0, i) * p_left.col(i).transpose(),  //
        p_right(1, i) * p_left.col(i).transpose(),  //
        p_right(2, i) * p_left.col(i).transpose();

  // The essential matrix lives in right null space of A.
  const Matrix<double, 9, 4> K =
      A.bdcSvd(Eigen::ComputeFullV).matrixV().rightCols(4);
  return K;
}

auto FivePointAlgorithmBase::reshape_null_space(
    const Matrix<double, 9, 4>& K) const -> std::array<Matrix3d, 4>
{
  const auto X = Map<const Matrix<double, 3, 3, RowMajor>>{K.col(0).data()};
  const auto Y = Map<const Matrix<double, 3, 3, RowMajor>>{K.col(1).data()};
  const auto Z = Map<const Matrix<double, 3, 3, RowMajor>>{K.col(2).data()};
  const auto W = Map<const Matrix<double, 3, 3, RowMajor>>{K.col(3).data()};

  return {X, Y, Z, W};
}

auto FivePointAlgorithmBase::essential_matrix_expression(
    const std::array<Matrix3d, 4>& null_space_bases) const
    -> Polynomial<Matrix3d>
{
  const auto& [X, Y, Z, W] = null_space_bases;
  return x * X + y * Y + z * Z + one_ * W;
}

auto FivePointAlgorithmBase::build_essential_matrix_constraints(
    const Polynomial<Matrix3d>& E,
    const std::array<Monomial, 20>& monomials) const -> Matrix<double, 10, 20>
{
  const auto EEt = E * E.t();
  const auto P = EEt * E - 0.5 * trace(EEt) * E;
  const auto Q = det(E);

  Matrix<double, 10, 20> A;
  A.setZero();

  // Save Q in the matrix.
  for (int j = 0; j < 20; ++j)
  {
    auto coeff = Q.coeffs.find(monomials[j]);
    if (coeff == Q.coeffs.end())
      continue;
    A(0, j) = coeff->second;
  }

  // Save P in the matrix.
  for (int a = 0; a < 3; ++a)
  {
    for (int b = 0; b < 3; ++b)
    {
      // N.B.: the row offset is 1.
      const auto i = 3 * a + b + 1;
      for (int j = 0; j < 20; ++j)
        A(i, j) = P(a, b).coeffs[monomials[j]];
    }
  }

  return A;
}


auto NisterFivePointAlgorithm::inplace_gauss_jordan_elimination(
    Matrix<double, 10, 20>& U) const -> void
{
  for (auto i = 0; i < 10; ++i)
  {
    U.row(i) /= U(i, i);
    for (auto j = i + 1; j < 10; ++j)
      U.row(j) = U.row(j) / U(j, i) - U.row(i);
  }

  for (auto i = 9; i >= 4; --i)
  {
    for (auto j = i - 1; j >= 4; --j)
    {
      U.row(j) = U.row(j) / U(j, i) - U.row(i);
      U.row(j) /= U(j, j);
    }
  }
}

auto NisterFivePointAlgorithm::form_resultant_matrix(
    const Matrix<double, 6, 10>& B_mat,
    UnivariatePolynomial<double> B[3][3]) const -> void
{
  auto to_poly = [this](const auto& row_vector) {
    auto p = Polynomial<double>{};
    for (int i = 0; i < row_vector.size(); ++i)
      p.coeffs[this->monomials[i + 10]] = row_vector[i];
    return p;
  };

  auto e = B_mat.row(0);
  auto f = B_mat.row(1);
  auto g = B_mat.row(2);
  auto h = B_mat.row(3);
  auto i = B_mat.row(4);
  auto j = B_mat.row(5);

  auto k = to_poly(e) - z * to_poly(f);
  auto l = to_poly(g) - z * to_poly(h);
  auto m = to_poly(i) - z * to_poly(j);

#ifdef SHOW_DEBUG_LOG
  SARA_DEBUG << "e = " << to_poly(e).to_string() << endl;
  SARA_DEBUG << "f = " << to_poly(f).to_string() << endl;
  SARA_DEBUG << "g = " << to_poly(g).to_string() << endl;
  SARA_DEBUG << "h = " << to_poly(h).to_string() << endl;
  SARA_DEBUG << "i = " << to_poly(i).to_string() << endl;
  SARA_DEBUG << "j = " << to_poly(j).to_string() << endl;

  SARA_DEBUG << "k = " << k.to_string() << endl;
  SARA_DEBUG << "l = " << l.to_string() << endl;
  SARA_DEBUG << "m = " << m.to_string() << endl;
#endif

  // 3. [x, y, 1]^T is a non-zero null vector in Null(B).
  B[0][0] = UnivariatePolynomial<double>{3};
  B[0][1] = UnivariatePolynomial<double>{3};
  B[0][2] = UnivariatePolynomial<double>{4};

  B[1][0] = UnivariatePolynomial<double>{3};
  B[1][1] = UnivariatePolynomial<double>{3};
  B[1][2] = UnivariatePolynomial<double>{4};

  B[2][0] = UnivariatePolynomial<double>{3};
  B[2][1] = UnivariatePolynomial<double>{3};
  B[2][2] = UnivariatePolynomial<double>{4};

  // 1st row.
  B[0][0][0] = k.coeffs[x];
  B[0][0][1] = k.coeffs[x * z];
  B[0][0][2] = k.coeffs[x * z.pow(2)];
  B[0][0][3] = k.coeffs[x * z.pow(3)];

  B[0][1][0] = k.coeffs[y];
  B[0][1][1] = k.coeffs[y * z];
  B[0][1][2] = k.coeffs[y * z.pow(2)];
  B[0][1][3] = k.coeffs[y * z.pow(3)];

  B[0][2][0] = k.coeffs[one_];
  B[0][2][1] = k.coeffs[z];
  B[0][2][2] = k.coeffs[z.pow(2)];
  B[0][2][3] = k.coeffs[z.pow(3)];
  B[0][2][4] = k.coeffs[z.pow(4)];

  // 2nd row.
  B[1][0][0] = l.coeffs[x];
  B[1][0][1] = l.coeffs[x * z];
  B[1][0][2] = l.coeffs[x * z.pow(2)];
  B[1][0][3] = l.coeffs[x * z.pow(3)];

  B[1][1][0] = l.coeffs[y];
  B[1][1][1] = l.coeffs[y * z];
  B[1][1][2] = l.coeffs[y * z.pow(2)];
  B[1][1][3] = l.coeffs[y * z.pow(3)];

  B[1][2][0] = l.coeffs[one_];
  B[1][2][1] = l.coeffs[z];
  B[1][2][2] = l.coeffs[z.pow(2)];
  B[1][2][3] = l.coeffs[z.pow(3)];
  B[1][2][4] = l.coeffs[z.pow(4)];

  // 3rd row.
  B[2][0][0] = m.coeffs[x];
  B[2][0][1] = m.coeffs[x * z];
  B[2][0][2] = m.coeffs[x * z.pow(2)];
  B[2][0][3] = m.coeffs[x * z.pow(3)];

  B[2][1][0] = m.coeffs[y];
  B[2][1][1] = m.coeffs[y * z];
  B[2][1][2] = m.coeffs[y * z.pow(2)];
  B[2][1][3] = m.coeffs[y * z.pow(3)];

  B[2][2][0] = m.coeffs[one_];
  B[2][2][1] = m.coeffs[z];
  B[2][2][2] = m.coeffs[z.pow(2)];
  B[2][2][3] = m.coeffs[z.pow(3)];
  B[2][2][4] = m.coeffs[z.pow(4)];

#ifdef SHOW_DEBUG_LOG
  SARA_DEBUG << "B00 = " << B[0][0] << endl;
  SARA_DEBUG << "B01 = " << B[0][1] << endl;
  SARA_DEBUG << "B02 = " << B[0][2] << endl;

  SARA_DEBUG << "B10 = " << B[1][0] << endl;
  SARA_DEBUG << "B11 = " << B[1][1] << endl;
  SARA_DEBUG << "B12 = " << B[1][2] << endl;

  SARA_DEBUG << "B20 = " << B[2][0] << endl;
  SARA_DEBUG << "B21 = " << B[2][1] << endl;
  SARA_DEBUG << "B22 = " << B[2][2] << endl;
#endif
}

auto NisterFivePointAlgorithm::solve_essential_matrix_constraints(
    const std::array<Matrix3d, 4>& E_bases,
    const Matrix<double, 10, 20>& A) const -> std::vector<EssentialMatrix>
{
  // Perform the Gauss-Jordan elimination on A and stop four rows earlier (cf.
  // paragraph 3.2.3).
#ifdef SHOW_DEBUG_LOG
  SARA_DEBUG << "Gauss-Jordan elimination..." << endl;
#endif
  Matrix<double, 10, 20> U = A;
  inplace_gauss_jordan_elimination(U);

  // Expand the determinant (cf. paragraph 3.2.4).
#ifdef SHOW_DEBUG_LOG
  SARA_DEBUG << "Determinant expansion..." << endl;
#endif
  const auto B_mat = U.bottomRightCorner(6, 10);
  UnivariatePolynomial<double> B[3][3];
  form_resultant_matrix(B_mat, B);

  const auto p0 = B[0][1] * B[1][2] - B[0][2] * B[1][1];
  const auto p1 = B[0][2] * B[1][0] - B[0][0] * B[1][2];
  const auto p2 = B[0][0] * B[1][1] - B[0][1] * B[1][0];

  const auto n = p0 * B[2][0] + p1 * B[2][1] + p2 * B[2][2];
#ifdef SHOW_DEBUG_LOG
  SARA_DEBUG << "n = " << n << endl;
#endif

  // Extract the roots of the polynomial using Jenkins-Traub rpoly solver,
  // instead of the two steps involving:
  //  (1) the SVD of the companion matrix
  //  (2) the root polishing using Sturm sequences.
  auto roots = decltype(rpoly(n)){};
  try
  {
#ifdef SHOW_DEBUG_LOG
    SARA_DEBUG << "Root extraction of " << n << endl;
#endif
    roots = rpoly(n);
  }
  catch (exception& e)
  {
    // And it's OK because some the 5 correspondences may be really wrong.
    SARA_DEBUG << "Polynomial solver failed: " << e.what() << endl;
  }

#ifdef SHOW_DEBUG_LOG
  SARA_DEBUG << "Extraction of xyz" << endl;
#endif

  auto xyzs = std::vector<Vector3d>{};
  for (const auto& z_complex : roots)
  {
    if (z_complex.imag() != 0)
      continue;

    const auto z = z_complex.real();

    const auto p0_z = p0(z);
    const auto p1_z = p1(z);
    const auto p2_z = p2(z);

    const auto x = p0_z / p2_z;
    const auto y = p1_z / p2_z;

    if (std::isnan(x) || std::isinf(x) || std::isnan(y) || std::isinf(y))
      continue;

    xyzs.push_back({x, y, z});
  }

  // 4. Build essential matrices for the real solutions.
#ifdef SHOW_DEBUG_LOG
  SARA_DEBUG << "Extraction of essential matrices..." << endl;
#endif
  auto Es = std::vector<EssentialMatrix>{};
  Es.reserve(10);

  for (const auto& xyz: xyzs)
  {
    auto E = EssentialMatrix{};
    E.matrix() = xyz[0] * E_bases[0] + xyz[1] * E_bases[1] +
                 xyz[2] * E_bases[2] + E_bases[3];
    Es.push_back(E);
  }

  return Es;
}

auto NisterFivePointAlgorithm::find_essential_matrices(
    const Matrix<double, 3, 5>& x1, const Matrix<double, 3, 5>& x2) const
    -> std::vector<EssentialMatrix>
{
  // 1. Extract the null space.
  const auto E_bases = extract_null_space(x1, x2);

  // 2. Form the epipolar constraints.
  const auto E_bases_reshaped = reshape_null_space(E_bases);
  const auto E_expr = essential_matrix_expression(E_bases_reshaped);
  const auto E_constraints = build_essential_matrix_constraints(E_expr);

  // 3. Solve the epipolar constraints.
  return solve_essential_matrix_constraints(E_bases_reshaped, E_constraints);
}


auto SteweniusFivePointAlgorithm::solve_essential_matrix_constraints(
    const Matrix<double, 9, 4>& E_bases, const Matrix<double, 10, 20>& M) const
    -> std::vector<EssentialMatrix>
{
  // This follows the Matlab code at the end of section 4. of "Recent
  // Developments on Direct Relative Orientation", Stewenius et al.

  Eigen::FullPivLU<Matrix10d> lu(M.block<10, 10>(0, 0));
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
  At.block<2, 10>(3, 0) = - B.block<2, 10>(4, 0);
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

    auto E = Matrix3d{};
    auto vec_E = Map<Matrix<double, 9, 1>>(E.data());
    vec_E = E_bases * U.col(s).tail<4>().real();

    Es.emplace_back(E.transpose());
  }

  return Es;
}

auto SteweniusFivePointAlgorithm::find_essential_matrices(
    const Matrix<double, 3, 5>& x1, const Matrix<double, 3, 5>& x2) const
    -> std::vector<EssentialMatrix>
{
  // 1. Extract the null space.
  const auto E_bases = extract_null_space(x1, x2);

  // 2. Form the epipolar constraints.
  const auto E_bases_reshaped = reshape_null_space(E_bases);
  const auto E_expr = essential_matrix_expression(E_bases_reshaped);
  const auto E_constraints = build_essential_matrix_constraints(E_expr);

  // 3. Solve the epipolar constraints.
  return solve_essential_matrix_constraints(E_bases, E_constraints);
}

} /* namespace DO::Sara */
