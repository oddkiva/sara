// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>


namespace DO::Sara {

  //! @brief Calculates the roots of the quadratic polynomial.
  //! Implemented as described in:
  //! http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
  template <typename T>
  inline auto compute_quadratic_roots(T a, T b, T c)
      -> std::array<std::complex<T>, 2>
  {
    const auto sqrt_delta = std::sqrt(std::complex<double>(b * b - 4 * a * c));

    if (b >= 0)
      return {(-b - sqrt_delta) / (2 * a),  //
              (2 * c) / (-b - sqrt_delta)};
    else
      return {(2 * c) / (-b + sqrt_delta),  //
              (-b + sqrt_delta) / (2 * a)};
  }

  template <typename T>
  inline auto compute_quadratic_real_roots(T a, T b, T c) -> std::array<T, 2>
  {
    const auto delta = b * b - 4 * a * c;
    if (delta < 0)
      return {
          std::numeric_limits<T>::quiet_NaN(),
          std::numeric_limits<T>::quiet_NaN(),
      };

    const auto sqrt_delta = std::sqrt(delta);
    if (b >= 0)
      return {(-b - sqrt_delta) / (2 * a),  //
              (2 * c) / (-b - sqrt_delta)};
    else
      return {(2 * c) / (-b + sqrt_delta),  //
              (-b + sqrt_delta) / (2 * a)};
  }

  template <typename T>
  void roots(const Univariate::UnivariatePolynomial<T, 2>& P,
             std::complex<T>& x1, std::complex<T>& x2, bool& real_roots)
  {
    const T& a = P[2];
    const T& b = P[1];
    const T& c = P[0];
    T delta = b * b - 4 * a * c;
    x1 = (-b - sqrt(std::complex<T>(delta))) / (2 * a);
    x2 = (-b + sqrt(std::complex<T>(delta))) / (2 * a);
    if (delta >= 0)
      real_roots = true;
    else
      real_roots = false;
  }

  // Discriminant precision: 1e-3.
  template <typename T>
  void roots(const Univariate::UnivariatePolynomial<T, 3>& P,
             std::complex<T>& z1, std::complex<T>& z2, std::complex<T>& z3,
             T eps = T(1e-3))
  {
    const auto a = static_cast<T>(1);
    const auto b = P[2] / P[3];
    const auto c = P[1] / P[3];
    const auto d = P[0] / P[3];
    static_assert(std::is_same_v<decltype(a), const T>);
    static_assert(std::is_same_v<decltype(b), const T>);
    static_assert(std::is_same_v<decltype(c), const T>);
    static_assert(std::is_same_v<decltype(d), const T>);

    // Cardano's formula.
    const auto p = (3 * c - b * b) / 3;
    const auto q = (-9 * c * b + 27 * d + 2 * b * b * b) / 27;
    const auto delta = q * q + 4 * p * p * p / 27;
    static_assert(std::is_same_v<decltype(p), const T>);
    static_assert(std::is_same_v<decltype(q), const T>);
    static_assert(std::is_same_v<decltype(delta), const T>);

    if (delta < -eps)
    {
      constexpr auto pi = static_cast<T>(M_PI);
      const auto theta = std::acos(-q / 2 * std::sqrt(27 / (-p * p * p))) / 3;
      z1 = 2 * std::sqrt(-p / 3) * std::cos(theta);
      z2 = 2 * std::sqrt(-p / 3) * std::cos(theta + 2 * pi / 3);
      z3 = 2 * std::sqrt(-p / 3) * std::cos(theta + 4 * pi / 3);
    }
    else if (delta <= eps)
    {
      z1 = 3 * q / p;
      z2 = z3 = -3 * q / (2 * p);
    }
    else
    {
      const auto j = std::complex<T>{-1 / T(2), std::sqrt(T(3)) / 2};

      const auto r1 = (-q + std::sqrt(delta)) / 2;
      const auto r2 = (-q - std::sqrt(delta)) / 2;
      constexpr auto one_third = 1 / T(3);
      const auto u =
          r1 < 0 ? -std::pow(-r1, one_third) : std::pow(r1, one_third);
      const auto v =
          r2 < 0 ? -std::pow(-r2, one_third) : std::pow(r2, one_third);

      z1 = u + v;
      z2 = j * u + std::conj(j) * v;
      z3 = j * j * u + std::conj(j * j) * v;
    }

    z1 -= b / (3 * a);
    z2 -= b / (3 * a);
    z3 -= b / (3 * a);
  }

  // Involves the precision of the cubic equation solver: (1e-3.)
  template <typename T>
  void roots(const Univariate::UnivariatePolynomial<T, 4>& P,
             std::complex<T>& z1, std::complex<T>& z2, std::complex<T>& z3,
             std::complex<T>& z4, T eps = T(1e-6))
  {
    const auto a4 = static_cast<T>(1);
    const auto a3 = P[3] / P[4];
    const auto a2 = P[2] / P[4];
    const auto a1 = P[1] / P[4];
    const auto a0 = P[0] / P[4];
    static_assert(std::is_same_v<decltype(a0), const T>);
    static_assert(std::is_same_v<decltype(a1), const T>);
    static_assert(std::is_same_v<decltype(a2), const T>);
    static_assert(std::is_same_v<decltype(a3), const T>);
    static_assert(std::is_same_v<decltype(a4), const T>);

    auto Q = Univariate::UnivariatePolynomial<T, 3>{};
    Q[3] = 1;
    Q[2] = -a2;
    Q[1] = a1 * a3 - 4 * a0;
    Q[0] = 4 * a2 * a0 - a1 * a1 - a3 * a3 * a0;

    auto y1 = std::complex<T>{};
    auto y2 = std::complex<T>{};
    auto y3 = std::complex<T>{};
    roots<T>(Q, y1, y2, y3, eps);

    auto yr = std::real(y1);
    auto yi = std::abs(std::imag(y1));
    if (yi > std::abs(std::imag(y2)))
    {
      yr = std::real(y2);
      yi = std::abs(std::imag(y2));
    }
    if (yi > std::abs(std::imag(y3)))
    {
      yr = std::real(y3);
      yi = std::abs(std::imag(y3));
    }

    const auto radicand = a3 * a3 / 4 - a2 + yr;
    const auto R = std::sqrt(radicand);
    auto D = std::complex<T>{};
    auto E = std::complex<T>{};

    if (abs(R) > 0)
    {
      D = std::sqrt(3 * a3 * a3 / 4 - R * R - 2 * a2 +
                    (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / (4 * R));
      E = std::sqrt(3 * a3 * a3 / 4 - R * R - 2 * a2 -
                    (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / (4 * R));
    }
    else
    {
      D = std::sqrt(3 * a3 * a3 / 4 - 2 * a2 + 2 * std::sqrt(yr * yr - 4 * a0));
      E = std::sqrt(3 * a3 * a3 / 4 - 2 * a2 - 2 * std::sqrt(yr * yr - 4 * a0));
    }

    z1 = (R + D) / static_cast<T>(2);
    z2 = (R - D) / static_cast<T>(2);
    z3 = (-R + E) / static_cast<T>(2);
    z4 = (-R - E) / static_cast<T>(2);

    // Check Viete's formula.
    /*double p = a2 - 3*a3*a3/8;
    double q = a1 - a2*a3/2 + a3*a3*a3/8;
    double r = a0 - a1*a3/4 + a2*a3*a3/16 - 3*a3*a3*a3*a3/256;

    cout << "-2p = " << -2*p << endl;
    cout << pow(z1,2) + pow(z2,2) + pow(z3,2) + pow(z4,2) << endl;
    cout << "-3*q = " << -3*q << endl;
    cout << pow(z1,3) + pow(z2,3) + pow(z3,3) + pow(z4,3) << endl;
    cout << "2p^2 - 4r = " << 2*p*p - 4*r << endl;
    cout << pow(z1,4) + pow(z2,4) + pow(z3,4) + pow(z4,4) << endl;
    cout << "5pq = " << 5*p*q << endl;
    cout << pow(z1,5) + pow(z2,5) + pow(z3,5) + pow(z4,5) << endl;*/

    z1 -= a3 / 4;
    z2 -= a3 / 4;
    z3 -= a3 / 4;
    z4 -= a3 / 4;
  }

}  // namespace DO::Sara
