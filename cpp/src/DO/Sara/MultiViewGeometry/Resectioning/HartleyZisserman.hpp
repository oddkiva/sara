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

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>


namespace DO::Sara {

  template <typename T, int M, int N>
  auto flipud(const Eigen::Matrix<T, M, N>& m)
  {
    auto m_flipped = Eigen::Matrix<T, M, N>{m.rows(), m.cols()};
    for (auto r = 0; r < m.rows(); ++r)
      m_flipped.row(r) = m.row(m.rows() - 1 - r);
    return m_flipped;
  }

  template <typename T, int M, int N>
  auto fliplr(const Eigen::Matrix<T, M, N>& m)
  {
    auto m_flipped = Eigen::Matrix<T, M, N>{m.rows(), m.cols()};
    for (auto c = 0; c < m.cols(); ++c)
      m_flipped.col(c) = m.col(m.cols() - 1 - c);
    return m_flipped;
  }

  /*!
   *  As described in:
   *  Appendix A4.1.1 in
   *  Multiple view Geometry in Computer Vision 2nd ed.
   *  by Richard Hartley and Andrew Zisserman.
   */
  template <typename T>
  auto rq_factorization_3x3(const Eigen::Matrix<T, 3, 3>& a,
                            Eigen::Matrix<T, 3, 3>& r,
                            Eigen::Matrix<T, 3, 3>& q)
  {
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    r = a;

    auto qx = Mat3{};
    auto qy = Mat3{};
    auto qz = Mat3{};

    auto norm = [](const auto x, const auto y) {
      return std::sqrt(x * x + y * y);
    };

    // Givens rotation about x axis.
    const auto nx = norm(r(2, 1), r(2, 2));
    const auto cx = r(2, 2) / nx;
    const auto sx = -r(2, 1) / nx;
    qx <<
      1,  0,   0,
      0, cx, -sx,
      0, sx,  cx;
    r *= qx;

    const auto ny = norm(r(2, 2), r(2, 0));
    const auto cy = r(2, 2) / nx;
    const auto sy = r(2, 0) / nx;
    qy <<
       cy, 0, sy,
        0, 1,  0,
      -sy, 0, cy;
    r *= qy;

    const auto nz = norm(r(1, 1), r(1, 0));
    const auto cz = r(1, 1) / nx;
    const auto sz = -r(1, 0) / nx;
    qz <<
       cz, -sz, 0,
       sz,  cz, 0,
        0,   0, 1;
    r *= qz;

    q = (qx * qy * qz).transpose();
  }

  template <typename T, int M, int N>
  auto rq_factorization(const Eigen::Matrix<T, M, N>& a,
                        Eigen::Matrix<T, M, N>& r,
                        Eigen::Matrix<T, M, N>& q)
  {
    if (a.rows() != a.cols())
      throw std::runtime_error{
          "The matrix must be square for the RQ factorization!"};

    const Eigen::Matrix<T, M, N> a1 = flipud(a).transpose();
    std::cout << "A1 =\n" << a1 << std::endl;
    const auto qr = a1.colPivHouseholderQr();

    q = qr.householderQ();
    r = qr.matrixQR().template triangularView<Eigen::Upper>();
    std::cout << "Q * R =\n" << q * r << std::endl;
    std::cout << "A1 - Q * R =\n" << a1 - q  * r << std::endl;

    // std::cout << "log |det(A1)| = " << qr.logAbsDeterminant() << std::endl;

    r.transposeInPlace();
    r = flipud(r);
    r = fliplr(r);

    q.transposeInPlace();
    q = flipud(q);

    // std::cout << "R =\n" << r << std::endl;
    // std::cout << "Q =\n" << q << std::endl;
    // std::cout << "Q.T * Q =\n" << q.transpose() * q << std::endl;
    // std::cout << "A - R * Q =\n" << a - r * q << std::endl;
  }

  template <typename T>
  auto resectioning_hartley_zisserman(const TensorView_<T, 2>& X,
                                      const TensorView_<T, 2>& x)
      -> PinholeCamera
  {
    if (X.rows() != x.rows())
      throw std::runtime_error{"X and x must have the same number of points!"};

    const Eigen::Matrix<T, 1, 4> zero_4 = Eigen::Matrix<T, 1, 4>::Zero();

    auto A = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>{X.rows() * 2, 12};
    for (auto r = 0; r < X.rows(); ++r)
    {
      const auto Xi_T = Eigen::Matrix<T, 1, 4>{X(r, 0), X(r, 1), X(r, 2), 1};
      const auto ui = x(r, 0);
      const auto vi = x(r, 1);

      A.row(2 * r + 0) << Xi_T, zero_4, -ui * Xi_T;
      A.row(2 * r + 1) << zero_4, Xi_T, -vi * Xi_T;
    }

    const auto svd = A.jacobiSvd(Eigen::ComputeFullV);
    const auto P_flat = svd.matrixV().col(11);

    auto P = Eigen::Matrix<T, 3, 4>{};
    P.row(0) = P_flat.template segment<4>(0);
    P.row(1) = P_flat.template segment<4>(4);
    P.row(2) = P_flat.template segment<4>(8);

    const Eigen::Matrix<T, 3, 3> M = P.template block<3, 3>(0, 0);

    auto K = Eigen::Matrix<T, 3, 3>{};
    auto R = Eigen::Matrix<T, 3, 3>{};
    auto t = Eigen::Matrix<T, 3, 1>{};

    // rq_factorization_3x3(M, K, R);
    rq_factorization(M, K, R);

    // Now flip the axes of K.
    const Eigen::Matrix<T, 3, 1> S = K.diagonal().array().sign().matrix();
    K = K * S.asDiagonal();
    R = S.asDiagonal() * R;

    t = K.inverse() * P.col(3);

    // Recover the scale of K.
    const auto scale = K(2, 2);
    K /= scale;

    return {K, R, t};
  }

}  // namespace DO::Sara
