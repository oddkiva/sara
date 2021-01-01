// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018-2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <Eigen/Eigen>


namespace DO::Sara {

  template <typename SquareMatrix>
  auto flipud(const SquareMatrix& m)
  {
    auto m_flipped = SquareMatrix{m.rows(), m.cols};
    for (auto r = 0; r < m.rows(); ++r)
      m_flipped.row(r) = m.row(m.rows() - 1 - m);
    return m_flipped;
  }

  template <typename SquareMatrix>
  auto rq_factorization(const SquareMatrix& m, SquareMatrix& r, SquareMatrix& q)
  {
    if (m.rows() != m.cols())
      throw std::runtime_error{
          "The matrix must be square for the RQ factorization!"};

    const auto m_flipped = flipud(m);

    const auto qr = m_flipped.colPivHouseholderQr();
    q = qr.householderQ();
    r = qr.matrixQR().triangularView<Eigen::Upper>();
    r = flipud(r);
  }

  template <typename T>
  auto resectioning_hartley_zisserman(const TensorView_<T, 2>& X,
                                      const TensorView_<T, 2>& x)
  {
    if (X.rows() != x.rows())
      throw std::runtime_error{"X and x must have the same number of points!"};

    const Matrix<T, 1, 4> zero_4 = Matrix<T, 1, 4>::Zero();

    auto A = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>{X.rows() * 2, 12};
    for (auto r = 0; r < X.rows(); ++r)
    {
      const auto Xi_T = Matrix<T, 1, 4>{X(r, 0), X(r, 1), X(r, 2), 1};
      const auto ui = x(r, 0);
      const auto vi = x(r, 1);

      A.row(2 * r + 0) << Xi_T, zero_4, -ui * Xi_T;
      A.row(2 * r + 1) << zero_4, Xi_T, -vi * Xi_T;
    }

    auto svd = A.jacobiSvd();
    const auto P_flat = svd.matrixV().col(11);
    auto P = Eigen::Matrix<T, 3, 4>{};
    P.row(0) = P_flat.segment(0, 4);
    P.row(1) = P_flat.segment(4, 8);
    P.row(2) = P_flat.segment(8, 12);

    const Matrix<T, 3, 3> M = P_reshaped.block<3, 3>(0, 0);

    auto K = Matrix<T, 3, 3>{};
    auto R = Matrix<T, 3, 3>{};
    auto t = Matrix<T, 3, 1>{};

    rq_factorization(M, R, K);
    t = K.inverse() * P.col(3);

    return {K, R, t}
  }


}  // namespace DO::Sara
