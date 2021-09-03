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
    using Mat34 = Eigen::Matrix<T, 3, 4>;

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
      solve_for_lambda();
      recover_all_poses();
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

    inline auto solve_for_lambda() -> void
    {
      // The first root is always real in this implementation.
      Mat3 D0 = D[0] + gamma[0] * D[1];

      eig3x3xknown0(D0, E, sigma);

      auto s = std::array<T, 2>{};
      const auto sp = std::sqrt(-sigma[1] / sigma[0]);
      const auto sm = -sp;

      const auto wm = calculate_w(sm);
      const auto wp = calculate_w(sp);

      const auto tau_m = solve_tau_quadratic_polynomial(wm);
      const auto tau_p = solve_tau_quadratic_polynomial(wp);

      for (const auto& tau : tau_m)
      {
        if (std::abs(std::imag(tau)) > 1e-10 && std::real(tau) < 0)
          continue;
        lambda_k.push_back(calculate_lambda_k(std::real(tau), wm));
      }

      for (const auto& tau : tau_p)
      {
        if (std::abs(std::imag(tau)) > 1e-10 && std::real(tau) < 0)
          continue;
        lambda_k.push_back(calculate_lambda_k(std::real(tau), wp));
      }


      // TODO: refine the lambda_k with Gauss-Newton polishing procedure.
    }

    inline auto recover_all_poses() -> void
    {
      for (const auto& lambda : lambda_k)
        pose_k.push_back(recover_pose(lambda));
    }

    inline auto get_eigen_vector(const Vec9& m, const T r) -> Vec3
    {
      const auto c = square(r) + m(0) * m(4) - r * (m(0) + m(4)) - square(m(1));
      const auto a1 = (r * m(2) + m(1) * m(5) - m(2) * m(4)) / c;
      const auto a2 = (r * m(5) + m(1) * m(2) - m(0) * m(5)) / c;
      const Vec3 v = Vec3{a1, a2, 1}.normalized();
      return v;
    }

    inline auto get_eigen_vector(const Mat3& m, const T r) -> Vec3
    {
      const auto c = square(r) + m(0, 0) * m(1, 1) -  //
                     r * (m(0, 0) * m(1, 1)) -        //
                     square(m(0, 1));
      const auto a1 = (r * m(0, 2) + m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) / c;
      const auto a2 = (r * m(1, 2) + m(0, 0) * m(0, 2) - m(0, 0) * m(1, 2)) / c;
      const Vec3 v = Vec3{a1, a2, 1}.normalized();
      return v;
    }

    inline auto eig3x3xknown0(const Mat3& M, Mat3& B, std::array<T, 2>& s)
        -> void
    {
      // lines 8-9
      const Vec3 b2 = M.col(1).cross(M.col(2)).normalized();

      // Ignore line 10 by not using the matrix M instead of its vectorized form
      // m = vec(M) = [M[0, :], M[1, :], M[2, :]].
      // We lose little in terms of readability.

      // Form the quadratic polynomial as described in lines 11-12.
      // but the original matrix M.
      auto p = Polynomial<T, 2>{};
#ifdef NON_VECTORIZED_MATRIX
      p[2] = 1;
      p[1] = -M(0, 0) - M(1, 1) - M(2, 2);
      p[0] = -square(M(0, 1)) - square(M(0, 2)) - square(M(1, 2)) +
             M(0, 0) * M(1, 1) + M(2, 2) + M(1, 2) * M(2, 2);
#else
      const Vec9 m = (Vec9{} << M.row(0), M.row(1), M.row(2)).finished();
      p[2] = 1;
      p[1] = -m(0) - m(4) - m(8);
      p[0] = -square(m(1)) - square(m(2)) - square(m(5)) + m(0) * m(4) + m(8) +
             m(4) * m(8);
#endif

      // Line 13: compute the real roots if they exist.
      auto are_reals = bool{};
      auto sigma_complex = std::array<std::complex<T>, 2>{};
      roots(p, sigma_complex[0], sigma_complex[1], are_reals);
      if (are_reals)
        sigma = {std::real(sigma_complex[0]), std::real(sigma_complex[1])};

        // Line 14-15.
#ifdef NON_VECTORIZED_MATRIX
      const auto b0 = get_eigen_vector(M, s[0]);
      const auto b1 = get_eigen_vector(M, s[1]);
#else
      const auto b0 = get_eigen_vector(m, s[0]);
      const auto b1 = get_eigen_vector(m, s[1]);
#endif

      // Line 16-19.
      if (std::abs(s[0]) > std::abs(s[1]))
      {
        // Line 16.
        B.col(0) = b0;
        B.col(1) = b1;
        B.col(2) = b2;
      }
      else
      {
        // Line 19
        B.col(0) = b1;
        B.col(1) = b0;
        B.col(2) = b2;
        std::swap(s[0], s[1]);
      }
    }

    inline auto calculate_w(T s) const -> std::array<T, 2>
    {
      const auto w0 = (E(1, 2) - s * E(1, 4)) / (s * E(0, 1) - E(0, 0));
      const auto w1 = (E(2, 0) - s * E(2, 1)) / (s * E(0, 1) - E(0, 0));
      return std::array{w0, w1};
    }

    inline auto solve_tau_quadratic_polynomial(const std::array<T, 2>& w) const
        -> std::array<T, 2>
    {
      // The τ-polynomial in tau arises from the quadratic form described in
      // Equation (14) of the paper.
      auto tau_polynomial = Polynomial<T, 2>{};
      // The coefficients of the τ-polynomial as shown in Equation (15) of the
      // paper.
      tau_polynomial[2] = (a(_02) - a(_01)) * square(w[0])        //
                          + 2 * a(_01) * b(_02) * w[1]            //
                          - a(_01);                               //
      tau_polynomial[1] = 2 * a(_01) - b(_02) * w[0]              //
                          - 2 * a(_02) * b(_01) * w[1]            //
                          - 2 * w[0] * w[1] * (a(_01) - a(_02));  //
      tau_polynomial[2] = (a(_02) - a(_01)) * square(w[0])        //
                          - 2 * a(_02) * b(_01) * w[0];           //

      auto tau = std::array<std::complex<T>, 2>{};
      auto real_roots = bool{};
      roots(tau_polynomial, tau[0], tau[1], real_roots);

      return tau;
    };

    inline auto calculate_lambda(const T tau, const std::array<T, 2>& w) -> Vec3
    {
      auto lambda = Vec3{};
      lambda(1) = std::sqrt(a(_12) / (tau * b(_12 + tau) + 1));
      lambda(2) = tau * lambda(1);
      lambda(0) = w[0] * lambda(1) + w[1] * lambda(2);
      return lambda;
    };

    inline auto recover_pose(const Vec3& lambda) -> Mat34
    {
      auto Y = Mat3{};
      Y.col(0) = lambda(0) * y.col(0) - lambda(1) * y.col(1);
      Y.col(1) = lambda(1) * y.col(1) - lambda(2) * y.col(2);
      Y.col(2) = Y.col(0).cross(Y.col(1));

      auto X = Mat3{};
      X(0) = x.col(0) - x.col(1);
      X(1) = x.col(1) - x.col(2);
      X(2) = X.col(0).cross(X.col(1));

      const Mat3 R = Y * X.inverse();
      const Vec3 t = lambda(0) * y(0) - R * x.col(0);

      auto pose = Mat34{};
      pose.leftCol(3) = R;
      pose.col(3) = t;

      return pose;
    };

    //! @brief 3 scene points with coordinates expressed in a reference frame.
    Mat3 x;
    //! @brief 3 backprojected light 3D rays expressed in the camera frame.
    Mat3 y;

    //! @brief Auxiliary variables: b12, b13, b23
    Vec3 b;
    //! @brief Auxiliary variables: a12, a13, a23
    Vec3 a;

    //! @brief The valid lambdas are in the 3 elliptic cylinders characterized
    //! by the 3 matrices M.
    std::array<Mat3, 3> M;

    //! @brief The valid lambdas are in the the 2 homogeneous 3D ellipses
    //! formed by the two 3x3 matrices D[0] and D[1].
    std::array<Mat3, 2> D;

    //! @brief The cubic polynomial formed by the linear combination of D[0] and
    //! D[1].
    Polynomial<T, 3> c;
    //! @brief The roots of the polynomial.
    std::array<std::complex<T>, 3> gamma;

    Mat3 E;
    std::array<T, 2> sigma;

    //! @brief The scales for each  backprojected rays.
    //! There are up to 4 possibles combinations of scales.
    std::vector<Vec3> lambda_k;

    //! @brief Recovered poses.
    std::vector<Mat34> pose_k;
  };

}  // namespace DO::Sara
