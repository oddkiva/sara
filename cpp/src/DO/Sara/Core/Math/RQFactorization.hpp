// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <Eigen/Eigen>

#include <iostream>


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

}  // namespace DO::Sara

namespace DO::Sara {

  template <typename  T>
  auto givens_rotation(const T& a, const T& b, T& c, T& s, T& r)
  {
    const auto h = std::hypot(a, b);
    if (b != 0)
    {
      const auto d = 1 / h;
      c = std::abs(a) * d;
      s = std::copysign(d, a) * b;
      r = std::copysign(1, a) * h;
    }
    else
    {
      c = 1;
      s = 0;
      r = a;
    }
  }

  //! This works better than the generic RQ fatorization.
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

    Mat3 qx = Mat3::Identity();
    Mat3 qy = Mat3::Identity();
    Mat3 qz = Mat3::Identity();

    // std::cout << "A =\n" << a << std::endl;

    // Givens rotation about x axis.
    const auto nx = std::hypot(r(2, 2), r(2, 1));
    if (nx > std::numeric_limits<T>::epsilon())
    {
      const auto cx = r(2, 2) / nx;
      const auto sx = -r(2, 1) / nx;
      qx <<
        1,  0,   0,
        0, cx, -sx,
        0, sx,  cx;
      r *= qx;
    }
    // std::cout << "Qx =\n" << qx << std::endl;

    const auto ny = std::hypot(r(2, 2), r(2, 0));
    if (ny > std::numeric_limits<T>::epsilon())
    {
      const auto cy = r(2, 2) / ny;
      const auto sy = r(2, 0) / ny;
      qy <<
         cy, 0, sy,
          0, 1,  0,
        -sy, 0, cy;
      r *= qy;
    }
    // std::cout << "Qy =\n" << qy << std::endl;

    const auto nz = std::hypot(r(1, 1), r(1, 0));
    if (nz > std::numeric_limits<T>::epsilon())
    {
      const auto cz = r(1, 1) / nz;
      const auto sz = -r(1, 0) / nz;
      qz <<
         cz, -sz, 0,
         sz,  cz, 0,
          0,   0, 1;
      r *= qz;
    }
    // std::cout << "Qz =\n" << qz << std::endl;

    // R = A * (Qx * Qy * Qz)
    // R * (Qx * Qy * Qz)^T

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
    const auto qr = a1.householderQr();

    q = qr.householderQ();
    r = qr.matrixQR().template triangularView<Eigen::Upper>();
    // std::cout << "R =\n" << r << std::endl;
    // std::cout << "Q =\n" << q << std::endl;
    // std::cout << "Q * R =\n" << q * r << std::endl;
    // std::cout << "A1 - Q * R =\n" << a1 - q  * r << std::endl;
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

}
