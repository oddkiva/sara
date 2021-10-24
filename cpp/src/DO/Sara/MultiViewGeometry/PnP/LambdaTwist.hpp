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
#include <DO/Sara/Core/Math/PolynomialRoots.hpp>
#include <DO/Sara/Core/Math/UsualFunctions.hpp>

#include <Eigen/Dense>

// #define EIGEN_IMPL
// #define DEBUG_LAMBDA_TWIST


namespace DO::Sara {

  //! @brief P3P solver.
  /*!
   *  This structure implements Lambda-Twist.
   *  https://openaccess.thecvf.com/content_ECCV_2018/papers/Mikael_Persson_Lambda_Twist_An_ECCV_2018_paper.pdf
   */
  template <typename T>
  struct LambdaTwist
  {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;
    using Mat34 = Eigen::Matrix<T, 3, 4>;

    //! @brief Indices which are zero-based.
    //!
    //! The paper mixes zero-based and one-based indices which makes it hard to
    //! follow at times.
    enum Index
    {
      _01 = 0,
      _02 = 1,
      _12 = 2
    };

    //! @brief Constructor.
    inline LambdaTwist(const Mat3& scene_points, const Mat3& backprojected_rays)
      : x{scene_points}
      , y{backprojected_rays}
    {
    }

    //! @brief Step 1/4.
    inline auto calculate_auxiliary_variables() -> void
    {
      // Calculate the distances between each scene points.
      a(_01) = (x.col(0) - x.col(1)).squaredNorm();
      a(_02) = (x.col(0) - x.col(2)).squaredNorm();
      a(_12) = (x.col(1) - x.col(2)).squaredNorm();

      // Calculate the cosine between the 3D rays.
      b(_01) = y.col(0).dot(y.col(1));
      b(_02) = y.col(0).dot(y.col(2));
      b(_12) = y.col(1).dot(y.col(2));

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

      // Form the conic equations.
      D[0] = M[_01] * a(_12) - M[_12] * a(_01);
      D[1] = M[_02] * a(_12) - M[_12] * a(_02);
    }

    //! @brief Step 2/4.
    inline auto solve_cubic_equation() -> void
    {
#define USE_PAPER_FORMULA
#ifdef USE_PAPER_FORMULA
      // N.B.: the paper wrongly swapped the coefficients c[2] and c[1].
      //
      // clang-format off
      c[3] = D[1].determinant();

      c[2] = D[0].col(0).dot(D[1].col(1).cross(D[1].col(2))) +
             // d11.T       (d22         x      d23)   +
             D[0].col(1).dot(D[1].col(2).cross(D[1].col(0))) +
             // d12.T       (d23         x      d21)   +
             D[0].col(2).dot(D[1].col(0).cross(D[1].col(1)));
             // d13.T       (d21         x      d22)

      c[1] = D[1].col(0).dot(D[0].col(1).cross(D[0].col(2))) +
             // d21.T        (d12        x      d13)         +
             D[1].col(1).dot(D[0].col(2).cross(D[0].col(0))) +
             // d22.T       (d13         x      d11)          +
             D[1].col(2).dot(D[0].col(0).cross(D[0].col(1)));
             // d23.T       (d11         x      d12)

      c[0] = D[0].determinant();
      // clang-format on
#else  // We also double-checked with SymPy: it is correct but not very readable
      const auto& D1 = D[0];
      const auto& D2 = D[1];
      c[3] = D2(0, 0) * D2(1, 1) * D2(2, 2) - D2(0, 0) * D2(1, 2) * D2(2, 1) -
             D2(0, 1) * D2(1, 0) * D2(2, 2) + D2(0, 1) * D2(1, 2) * D2(2, 0) +
             D2(0, 2) * D2(1, 0) * D2(2, 1) - D2(0, 2) * D2(1, 1) * D2(2, 0);
      c[2] = D1(0, 0) * D2(1, 1) * D2(2, 2) - D1(0, 0) * D2(1, 2) * D2(2, 1) -
             D1(0, 1) * D2(1, 0) * D2(2, 2) + D1(0, 1) * D2(1, 2) * D2(2, 0) +
             D1(0, 2) * D2(1, 0) * D2(2, 1) - D1(0, 2) * D2(1, 1) * D2(2, 0) -
             D1(1, 0) * D2(0, 1) * D2(2, 2) + D1(1, 0) * D2(0, 2) * D2(2, 1) +
             D1(1, 1) * D2(0, 0) * D2(2, 2) - D1(1, 1) * D2(0, 2) * D2(2, 0) -
             D1(1, 2) * D2(0, 0) * D2(2, 1) + D1(1, 2) * D2(0, 1) * D2(2, 0) +
             D1(2, 0) * D2(0, 1) * D2(1, 2) - D1(2, 0) * D2(0, 2) * D2(1, 1) -
             D1(2, 1) * D2(0, 0) * D2(1, 2) + D1(2, 1) * D2(0, 2) * D2(1, 0) +
             D1(2, 2) * D2(0, 0) * D2(1, 1) - D1(2, 2) * D2(0, 1) * D2(1, 0);
      c[1] = D1(0, 0) * D1(1, 1) * D2(2, 2) - D1(0, 0) * D1(1, 2) * D2(2, 1) -
             D1(0, 0) * D1(2, 1) * D2(1, 2) + D1(0, 0) * D1(2, 2) * D2(1, 1) -
             D1(0, 1) * D1(1, 0) * D2(2, 2) + D1(0, 1) * D1(1, 2) * D2(2, 0) +
             D1(0, 1) * D1(2, 0) * D2(1, 2) - D1(0, 1) * D1(2, 2) * D2(1, 0) +
             D1(0, 2) * D1(1, 0) * D2(2, 1) - D1(0, 2) * D1(1, 1) * D2(2, 0) -
             D1(0, 2) * D1(2, 0) * D2(1, 1) + D1(0, 2) * D1(2, 1) * D2(1, 0) +
             D1(1, 0) * D1(2, 1) * D2(0, 2) - D1(1, 0) * D1(2, 2) * D2(0, 1) -
             D1(1, 1) * D1(2, 0) * D2(0, 2) + D1(1, 1) * D1(2, 2) * D2(0, 0) +
             D1(1, 2) * D1(2, 0) * D2(0, 1) - D1(1, 2) * D1(2, 1) * D2(0, 0);
      c[0] = D1(0, 0) * D1(1, 1) * D1(2, 2) - D1(0, 0) * D1(1, 2) * D1(2, 1) -
             D1(0, 1) * D1(1, 0) * D1(2, 2) + D1(0, 1) * D1(1, 2) * D1(2, 0) +
             D1(0, 2) * D1(1, 0) * D1(2, 1) - D1(0, 2) * D1(1, 1) * D1(2, 0);
#endif

      // Solve the cubic polynomial.
      c /= c[3];
      compute_cubic_real_roots(c, gamma[0], gamma[1], gamma[2]);
    }

    //! @brief Step 3/4.
    inline auto solve_for_lambda() -> void
    {
      // The first root is always real in this implementation.
      const Mat3 D0 = D[0] + gamma[0] * D[1];
#ifdef DEBUG_LAMBDA_TWIST
      SARA_DEBUG << "D0 =\n" << D0 << std::endl;
      SARA_DEBUG << "det(D0) = " << D0.determinant() << std::endl;
#endif

      eig3x3known0(D0, E, sigma);
#ifdef DEBUG_LAMBDA_TWIST
      SARA_DEBUG << "E =\n" << E << std::endl;
      SARA_DEBUG << "sigma = " << sigma.transpose() << std::endl;
      SARA_CHECK((D0 - E * sigma.asDiagonal() * E.transpose()).norm());
      SARA_CHECK(E.colwise().norm());
      SARA_CHECK(E.determinant());
#endif

      const auto sp = std::sqrt(-sigma[1] / sigma[0]);
      const auto sm = -sp;
#ifdef DEBUG_LAMBDA_TWIST
      SARA_CHECK(sp);
      SARA_CHECK(sm);
#endif

      const auto wm = calculate_w(sm);
      const auto wp = calculate_w(sp);
#ifdef DEBUG_LAMBDA_TWIST
      SARA_CHECK(wm[0]);
      SARA_CHECK(wm[1]);
      SARA_CHECK(wp[0]);
      SARA_CHECK(wp[1]);
#endif

      const auto tau_m = solve_tau_quadratic_polynomial(wm);
      const auto tau_p = solve_tau_quadratic_polynomial(wp);

      for (const auto& tau : tau_m)
      {
        if (std::isnan(tau))
          continue;
        if (tau <= 0)
          continue;

        const auto lambda = calculate_lambda(tau, wm);
        if (!(lambda.array() > 0).all())
          continue;

        lambda_k.push_back(lambda);
      }

      for (const auto& tau : tau_p)
      {
        if (std::isnan(tau))
          continue;
        if (tau <= 0)
          continue;

        const auto lambda = calculate_lambda(tau, wp);
        if (!(lambda.array() > 0).all())
          continue;

        lambda_k.push_back(lambda);
      }
    }

    //! @brief Step 4/4.
    inline auto recover_all_poses() -> void
    {
      for (const auto& lambda : lambda_k)
        pose_k.push_back(recover_pose(lambda));
    }

    //! @brief Calculate the eigen decomposition of a 3x3 matrix with a zero
    //! eigenvalue as described in the paper.
    /*!
     *  We don't follow the paper line by line. The paper basically follows the
     *  following approach:
     *  1. Calculate the eigenvalues by solving the characteristic polynomial.
     *  2. Calculate the eigenvectors from the eigenvalues.
     *
     *  This is part of step 3/4.
     */
    static inline auto eig3x3known0(const Mat3& M, Mat3& B, Vec3& s) -> void
    {
#ifdef DEBUG_LAMBDA_TWIST
      print_stage("eig3x3known0");
#endif

#if defined(EIGEN_IMPL)
      // More robust, much simpler and also direct.
      // Might be slower, but this should be acceptable.
      auto es = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>{};
      es.computeDirect(M);
#  ifdef DEBUG_LAMBDA_TWIST
      std::cout << "Eigenvalues = " << es.eigenvalues().transpose()
                << std::endl;
      std::cout << "Eigenvectors = " << std::endl
                << es.eigenvectors() << std::endl;
#  endif

      // The first eigenvalue is always negative, the second is zero, and the
      // third one is positive.
      // The right-handedness is preserved if we rotate the column-vectors.
      B.col(0) = es.eigenvectors().col(2);
      B.col(1) = es.eigenvectors().col(0);
      B.col(2) = es.eigenvectors().col(1);
#  ifdef DEBUG_LAMBDA_TWIST
      SARA_CHECK(B.determinant());
#  endif

      s(0) = es.eigenvalues()(2);
      s(1) = es.eigenvalues()(0);
      s(2) = es.eigenvalues()(1);
#else  // MINE
      // Calculate the eigen values by solving the polynomial characteristic in
      // dimension 3:
      //
      //   X^3 - tr(M) X^2 + ((tr(M)^2 - tr(M^2)) / 2) X + det(M) = 0
      //
      // This formula is found in e.g.:
      // - https://mathworld.wolfram.com/CharacteristicPolynomial.html
      //
      // The third eigenvalue is always 0, so let's set it without further ado:
      s(2) = 0;

      // Because one eigenvalue is 0, so det(M) = 0, we just need to solve the
      // quadratic polynomial to find the two remaining eigenvalues.
      //
      //   X^2 - tr(M) X + ((tr(M)^2 - tr(M^2)) / 2) = 0
      //
      auto p = UnivariatePolynomial<T, 2>{};
      p[2] = 1;
      p[1] = -M.trace();
      // (tr(M)^2 - tr(M^2)) / 2 has the following analytical expression.
      p[0] = (M(0, 0) * M(1, 1) - square(M(0, 1))) +
             (M(0, 0) * M(2, 2) - square(M(0, 2))) +
             (M(1, 1) * M(2, 2) - square(M(1, 2)));

      // Compute the remaining nonzero eigenvalues (cf. line 13)
      if (!compute_quadratic_real_roots(p, s(0), s(1)))
        throw std::runtime_error{"The roots must be real!"};

      // Let's swap the eigenvalues as described in the paper once
      // for all. (cf. Line 16-19)
      //
      // The first eigenvalue is positive and the second one is negative.
      // So swap the eigenvalues if necessary.
      if (s[0] < s[1])
        std::swap(s(0), s(1));

      // Now we are following D. Eberly's paper instead of the method proposed
      // in the paper:
      //
      // - https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
      //
      // I have found that method in the paper is not robust numerically.

      auto compute_orthogonal_complement = [](Mat3& M) -> void {
        const auto W = M.col(0);
        auto U = M.col(1);
        auto V = M.col(2);

        if (std::abs(W(0)) > std::abs(W(1)))
        {
          const auto inverse_length =
              1 / std::sqrt(square(W(0)) + square(W(2)));
          U << -W(2) * inverse_length, 0, W(0) * inverse_length;
        }
        else
        {
          const auto inverse_length =
              1 / std::sqrt(square(W(1)) + square(W(2)));
          U << 0, W(2) * inverse_length, -W(1) * inverse_length;
        }

        V = W.cross(U);
      };

      auto compute_eigen_vector_0 = [](const Mat3& M, const T eigval) -> Vec3 {
        const Mat3 A = M - eigval * Mat3::Identity();
        const auto ri_x_rj = std::array<Vec3, 3>{A.row(0).cross(A.row(1)),
                                                 A.row(0).cross(A.row(2)),
                                                 A.row(1).cross(A.row(2))};

        const auto d_ij =
            std::array{ri_x_rj[0].squaredNorm(), ri_x_rj[1].squaredNorm(),
                       ri_x_rj[2].squaredNorm()};

        const auto best_index =
            std::max_element(d_ij.begin(), d_ij.end()) - d_ij.begin();
#  ifdef DEBUG_LAMBDA_TWIST
        SARA_CHECK(best_index);
#  endif

        const Vec3 eigvec0 = ri_x_rj[best_index] / std::sqrt(d_ij[best_index]);
#  ifdef DEBUG_LAMBDA_TWIST
        SARA_CHECK(eigvec0.squaredNorm());
#  endif

        return eigvec0;
      };

      auto compute_eigen_vector_1 = [](const Mat3& A, const Mat3& B,
                                       T eval1) -> Vec3 {
        const auto U = B.col(1);
        const auto V = B.col(2);

        const auto AU = A * B.col(1);
        const auto AV = A * B.col(2);

        // J = [U, V]
        // Calculate M = J.T @ (A - eval1 * I)  @ J
        // M is a 2x2 symmetric matrix where.
        auto m00 = U.dot(AU) - eval1;
        auto m01 = V.dot(AU);
        auto m11 = V.dot(AV) - eval1;

        const auto abs_m00 = std::abs(m00);
        const auto abs_m01 = std::abs(m01);
        const auto abs_m11 = std::abs(m11);

        if (abs_m00 > abs_m11)
        {
          const auto abs_max_comp = std::max(abs_m00, abs_m01);
          if (abs_max_comp > 0)
          {
            if (abs_m00 >= abs_m01)
            {
              m01 /= m00;
              m00 = 1 / std::sqrt(1 + square(m01));
              m01 *= m00;
            }
            else
            {
              m00 /= m01;
              m01 = 1 / std::sqrt(1 + square(m00));
              m00 *= m01;
            }
            return m01 * U - m00 * V;
          }
          else
            return U;
        }
        else
        {
          const auto abs_max_comp = std::max(abs_m11, abs_m01);
          if (abs_max_comp > 0)
          {
            if (abs_m11 >= abs_m01)
            {
              m01 /= m11;
              m11 = 1 / std::sqrt(1 + square(m01));
              m01 *= m11;
            }
            else
            {
              m11 /= m01;
              m01 = 1 / std::sqrt(1 + square(m11));
              m11 *= m01;
            }
            return m11 * U - m01 * V;
          }
          else
            return U;
        }
      };

      // We calculate the 1st and 2nd eigenvectors (cf. line 14-15).
      // and directly the eigen vectors (we don't need the check in line 16).
      B.col(0) = compute_eigen_vector_0(M, s(0));

      compute_orthogonal_complement(B);
#  ifdef DEBUG_LAMBDA_TWIST
      SARA_DEBUG << "[Orthgonal complement] B =\n" << B << std::endl;
#  endif

      B.col(1) = compute_eigen_vector_1(M, B, s(1));

      // We don't follow line 8-9 to calculate the 3rd eigenvector.
      // Instead, let's just use the cross product.
      B.col(2) = B.col(0).cross(B.col(1));

#  ifdef DEBUG_LAMBDA_TWIST
      SARA_DEBUG << "B =\n" << B << std::endl;
#  endif
#endif
    }

    //! @brief Calculate the linear combination which relates the scales:
    //! λ[0] = w[0] * λ[1] + w[1] * λ[2].
    //!
    //! This is part of step 3/4.
    inline auto calculate_w(T s) const -> std::array<T, 2>
    {
#ifdef DEBUG_LAMBDA_TWIST
      print_stage("Calculate w");
      SARA_DEBUG << "E =\n" << E  << std::endl;
#endif

      const auto& e0 = E(0, 0); const auto& e1 = E(0, 1);
      const auto& e3 = E(1, 0); const auto& e4 = E(1, 1);
      const auto& e6 = E(2, 0); const auto& e7 = E(2, 1);

      const auto w0 = (e3 - s * e4) / (s * e1 - e0);
      const auto w1 = (e6 - s * e7) / (s * e1 - e0);

      return {w0, w1};
    }

    //! @brief Solve the auxiliary scale τ that relates the scales:
    //! λ[1] = τ * λ[2].
    //!
    //! This is part of step 3/4.
    inline auto solve_tau_quadratic_polynomial(const std::array<T, 2>& w) const
        -> std::array<T, 2>
    {
#ifdef DEBUG_LAMBDA_TWIST
      print_stage("Calculate tau roots");
#endif

      // The τ-polynomial in tau arises from the quadratic form described in
      // Equation (14) of the paper.
      auto tau_polynomial = UnivariatePolynomial<T, 2>{};

      // The coefficients of the τ-polynomial as shown in Equation (15) of the
      // paper are wrong.
      // After calculating by hand and double-checking the formula with SymPy,
      // this is the right formula.
      const auto& w0 = w[0];
      const auto& w1 = w[1];
      const auto& a01 = a(_01);
      const auto& a12 = a(_12);
      const auto& b01 = b(_01);
      const auto& b12 = b(_12);
      tau_polynomial[2] = -a01 + a12 * square(w1);
      tau_polynomial[1] = 2 * a01 * b12         //
                          - 2 * a12 * b01 * w1  //
                          + 2 * a12 * w0 * w1;
      tau_polynomial[0] = -a01 - 2 * a12 * b01 * w0 + a12 * square(w0) + a12;
#ifdef DEBUG_LAMBDA_TWIST
      SARA_CHECK(tau_polynomial);
#endif

      // Normalize the polynomial to its monic equivalent.
      tau_polynomial /= tau_polynomial[2];
      auto tau = std::array<T, 2>{};
      if (!compute_quadratic_real_roots(tau_polynomial, tau[0], tau[1]))
        std::fill(tau.begin(), tau.end(), std::numeric_limits<T>::quiet_NaN());

#ifdef DEBUG_LAMBDA_TWIST
      SARA_CHECK(tau[0]);
      SARA_CHECK(tau[1]);
      SARA_CHECK(tau_polynomial(tau[0]));
      SARA_CHECK(tau_polynomial(tau[1]));
#endif

      return tau;
    };

    //! @brief Determine the scales λ from the auxiliary variables.
    //!
    //! This is part of step 3/4.
    inline auto calculate_lambda(const T tau, const std::array<T, 2>& w) -> Vec3
    {
#ifdef DEBUG_LAMBDA_TWIST
      print_stage("Calculate lambda (the distances)");
#endif

      auto lambda = Vec3{};
      lambda(1) = std::sqrt(a(_12) / (square(tau) - 2 * b(_12) * tau + 1));
      lambda(2) = tau * lambda(1);
      lambda(0) = w[0] * lambda(1) + w[1] * lambda(2);

#ifdef DEBUG_LAMBDA_TWIST
      SARA_DEBUG << "ARE THE LAMBDA CORRECT?" << std::endl;
      SARA_CHECK(tau);
      SARA_CHECK(lambda.transpose());
      SARA_CHECK(lambda.transpose() * D[0] * lambda);
      SARA_CHECK(lambda.transpose() * D[1] * lambda);
      SARA_CHECK(lambda.transpose() * M[_01] * lambda - a(_01));
      SARA_CHECK(lambda.transpose() * M[_02] * lambda - a(_02));
      SARA_CHECK(lambda.transpose() * M[_12] * lambda - a(_12));
#endif

      return lambda;
    };

    //! @brief Recover the candidate poses from the candidate scales λ.
    //! This is part of step 4/4.
    inline auto recover_pose(const Vec3& lambda) -> Mat34
    {
      auto Y = Mat3{};
      Y.col(0) = lambda(0) * y.col(0) - lambda(1) * y.col(1);
      Y.col(1) = lambda(1) * y.col(1) - lambda(2) * y.col(2);
      Y.col(2) = Y.col(0).cross(Y.col(1));

      auto X = Mat3{};
      X.col(0) = x.col(0) - x.col(1);
      X.col(1) = x.col(1) - x.col(2);
      X.col(2) = X.col(0).cross(X.col(1));

      const Mat3 R = Y * X.inverse();
      const Vec3 t = lambda(0) * y.col(0) - R * x.col(0);

      const auto pose = (Mat34{} << R, t).finished();

#ifdef DEBUG_LAMBDA_TWIST
      const Eigen::Matrix<T, 3, 3> Xc = (pose * x.colwise().homogeneous());
      SARA_DEBUG << "Recovered R =\n" << R << std::endl;
      SARA_DEBUG << "Recovered t = " << t.transpose() << std::endl;
      SARA_DEBUG << "Xc = camera coordinates =\n" << Xc << std::endl;
      SARA_DEBUG << "Pose =\n" << pose << std::endl;
#endif

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
    UnivariatePolynomial<T, 3> c;
    //! @brief The roots of the polynomial.
    std::array<T, 3> gamma;

    Mat3 E;
    Vec3 sigma;

    //! @brief The scales for each  backprojected rays.
    //! There are up to 4 possibles combinations of scales.
    std::vector<Vec3> lambda_k;

    //! @brief Recovered poses.
    std::vector<Mat34> pose_k;
  };


  //! @brief Solve the P3P problem using Lambda-Twist.
  template <typename T>
  inline auto solve_p3p(const Eigen::Matrix<T, 3, 3>& scene_points,
                        const Eigen::Matrix<T, 3, 3>& backprojected_rays)
      -> std::vector<Eigen::Matrix<T, 3, 4>>
  {
    auto lt = LambdaTwist<T>{scene_points, backprojected_rays};
    lt.calculate_auxiliary_variables();
    lt.solve_cubic_equation();
    lt.solve_for_lambda();
    lt.recover_all_poses();
    return lt.pose_k;
  }

}  // namespace DO::Sara
