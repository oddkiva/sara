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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Math/UsualFunctions.hpp>
#include <DO/Sara/Geometry/Tools/PolynomialRoots.hpp>

#include <Eigen/Dense>


namespace DO::Sara {

  template <typename T>
  struct LambdaTwist
  {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Vec9 = Eigen::Matrix<T, 9, 1>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    enum Index
    {
      _01 = 0,
      _02 = 1,
      _12 = 2
    };

    inline LambdaTwist(const Mat3& scene_points, const Mat3& backprojected_rays)
      : x{scene_points}
      , y{backprojected_rays}
    {
      calculate_auxiliary_variables();
      solve_cubic_polynomial();
    }

    inline auto calculate_auxiliary_variables() -> void
    {
      b(_01) = y.col(0).dot(y.col(1));
      b(_02) = y.col(0).dot(y.col(2));
      b(_12) = y.col(1).dot(y.col(2));
      SARA_CHECK(b.transpose());

      a(_01) = (x.col(0) - x.col(1)).squaredNorm();
      a(_02) = (x.col(0) - x.col(2)).squaredNorm();
      a(_12) = (x.col(1) - x.col(2)).squaredNorm();
      SARA_CHECK(a.transpose());

      // clang-format off
      M[_01] <<
              1, -b(_01), 0,
        -b(_01),       1, 0,
              0,       0, 0;

      M[_02] <<
              1, 0, -b(_02),
              0, 0,       0,
        -b(_02), 0,       1;

      M[_12] <<
              0,      0,       0,
              0,      1, -b(_12),
              0, -b(_12),      1;
      // clang-format on

      SARA_DEBUG << "M01 =\n" << M[_01] << std::endl;
      SARA_DEBUG << "M02 =\n" << M[_02] << std::endl;
      SARA_DEBUG << "M12 =\n" << M[_12] << std::endl;

      // Form the conical equations.
      D[0] = M[_01] * a(_12) - M[_12] * a(_01);
      D[1] = M[_02] * a(_12) - M[_12] * a(_02);
      SARA_DEBUG << "D0 =\n" << D[0] << std::endl;
      SARA_DEBUG << "D1 =\n" << D[1] << std::endl;
    }

    inline auto solve_cubic_polynomial() -> void
    {
      // clang-format off
      c[3] = D[1].determinant();

      c[2] = D[1].col(0).dot(D[0].col(1).cross(D[0].col(2))) +
             // d21.T        (d12        x      d13)         +
             D[1].col(1).dot(D[0].col(2).cross(D[0].col(0))) +
             // d22.T       (d13         x      d11)          +
             D[1].col(2).dot(D[0].col(0).cross(D[0].col(1)));
             // d23.T       (d11         x      d12)

      c[1] = D[0].col(0).dot(D[1].col(1).cross(D[1].col(2))) +
             // d11.T       (d22         x      d23)   +
             D[0].col(1).dot(D[1].col(2).cross(D[1].col(0))) +
             // d12.T       (d23         x      d21)   +
             D[0].col(2).dot(D[1].col(0).cross(D[1].col(1)));
             // d13.T       (d21         x      d22)

      c[0] = D[0].determinant();
      // clang-format on

      // Solve the cubic polynomial.
      roots(c, gamma[0], gamma[1], gamma[2]);
    }

    auto solve_for_lambda() -> void
    {
      // The first root is always real in this implementation.
      Mat3 D0 = D[0] + gamma[0] * D[1];

      eig3x3xknown0(D0, E, sigma);

      const auto sp = std::sqrt(-sigma[1] / sigma[0]);
      const auto sm = -std::sqrt(-sigma[1] / sigma[0]);
    }

    auto get_eigen_vector(const Vec9& m, const T r) -> Vec3
    {
      const auto c = square(r) + m(0) * m(4) - r * (m(0) + m(4)) - square(m(1));
      const auto a1 = (r * m(2) + m(1) * m(5) - m(2) * m(4)) / c;
      const auto a2 = (r * m(5) + m(1) * m(2) - m(0) * m(5)) / c;
      const Vec3 v = Vec3{a1, a2, 1}.normalized();
      return v;
    }

    auto eig3x3xknown0(const Mat3& M, Mat3& B, std::array<T, 2>& s) -> void
    {
      const Vec3 b2 = M.col(1).cross(M.col(2)).normalized();
      const auto m = Eigen::Map<const Vec3>{M.data(), M.size()};

      auto p = Polynomial<T, 2>{};
      p[2] = 1;
      p[1] = m(0) - m(4) - m(8);
      p[0] = -square(m(1)) - square(m(2)) - square(m(5)) + m(0) * m(4) + m(8) +
             m(4) * m(8);
      auto are_reals = bool{};
      auto sigma_complex = std::array<std::complex<T>, 2>{};
      roots(p, sigma_complex[0], sigma_complex[1], are_reals);
      if (are_reals)
        sigma = {std::real(sigma_complex[0]), std::real(sigma_complex[1])};

      const auto b0 = get_eigen_vector(m, s[0]);
      const auto b1 = get_eigen_vector(m, s[1]);

      if (std::abs(s[0]) > std::abs(s[1]))
      {
        B.col(0) = b0;
        B.col(1) = b1;
        B.col(2) = b2;
      }
      else
      {
        B.col(0) = b1;
        B.col(1) = b0;
        B.col(2) = b2;
        std::swap(s[0], s[1]);
      }
    }

    //! @brief 3 scene points with coordinates expressed in a reference frame.
    Mat3 x;
    //! @brief 3 backprojected light 3D rays expressed in the camera frame.
    Mat3 y;

    //! @brief Auxiliary variables: b12, b13, b23
    Vec3 b;
    //! @brief Auxiliary variables: a12, a13, a23
    Vec3 a;

    std::array<Mat3, 3> M;
    std::array<Mat3, 2> D;

    //! The scales for each  backprojected rays.
    Vec3 lambda;

    //! @brief The cubic polynomial.
    Polynomial<T, 3> c;
    //! @brief The roots of the polynomial.
    std::array<std::complex<T>, 3> gamma;

    Mat3 E;
    std::array<T, 2> sigma;
  };

}  // namespace DO::Sara
