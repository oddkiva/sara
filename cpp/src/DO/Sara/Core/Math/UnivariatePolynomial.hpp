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

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


namespace DO::Sara::Univariate {

  //! @ingroup Core
  //! @defgroup Math Some mathematical tools
  //! @{

  template <typename T, int N = -1>
  class UnivariatePolynomial;

  //! @brief Univariate polynomial class with degree known at compile time.
  template <typename T, int N>
  class UnivariatePolynomial
  {
  public:
    using coefficient_type = T;

    //! @{
    //! Constructors.
    inline UnivariatePolynomial() = default;

    inline explicit UnivariatePolynomial(const T* coeff)
    {
      std::copy(coeff, coeff + N + 1, _coeff);
    }

    inline UnivariatePolynomial(std::initializer_list<T> list)
    {
      std::copy(list.begin(), list.end(), _coeff.begin());
    }

    inline UnivariatePolynomial(const UnivariatePolynomial& P)
    {
      copy(P);
    }
    //! @}

    //! Assign a new polynomial to the polynomial object.
    inline auto operator=(const UnivariatePolynomial& P)
        -> UnivariatePolynomial&
    {
      copy(P);
      return *this;
    }

    //! @brief Return the degree of the polynomial.
    inline constexpr auto degree() const -> int
    {
      return N;
    }

    //! @{
    //! Return the polynomial coefficient at degree 'i'.
    inline auto operator[](int degree) -> coefficient_type&
    {
      return _coeff[degree];
    }

    inline auto operator[](int degree) const -> const coefficient_type&
    {
      return _coeff[degree];
    }
    //! @}

    //! @brief Evaluate polynomial at point 'x' using Horner's method
    //! evaluation.
    template <typename U>
    auto operator()(U x) const -> decltype(coefficient_type{} + U{})
    {
      if (x == U{})
        return _coeff[0];

      using result_type = decltype(coefficient_type{} + U{});
      auto b = result_type(_coeff[degree()]);
      for (auto i = 1u; i < _coeff.size(); ++i)
        b = _coeff[degree() - i] + b * x;
      return b;
    }

    //! @{
    //! Comparison operator.
    inline auto operator==(const UnivariatePolynomial& other) const -> bool
    {
      for (int i = 0; i <= N; ++i)
        if (_coeff[i] != other._coeff[i])
          return false;
      return true;
    }

    inline auto operator!=(const UnivariatePolynomial& other) const -> bool
    {
      return !operator=(other);
    }
    //! @}

    //! I/O.
    //! @{
    auto to_string() const -> std::string
    {
      auto str = std::string{};
      std::ostringstream oss;
      for (int i = degree(); i >= 0; --i)
      {
        if (_coeff[i] >= 0)
          oss << "+";
        else
          oss << "-";
        oss << std::abs(_coeff[i]);
        if (i > 0)
          oss << "X^" << i << " ";
      }
      return oss.str();
    }

    friend inline auto operator<<(std::ostream& os,
                                  const UnivariatePolynomial& p)
        -> std::ostream&
    {
      os << p.to_string();
      return os;
    }
    //! @}

  private:
    inline auto copy(const UnivariatePolynomial& other) -> void
    {
      std::copy(other._coeff.begin(), other._coeff.end(), _coeff.begin());
    }

  private:
    std::array<T, N + 1> _coeff;
  };

  //! @brief Univariate polynomial class with degree known at runtime.
  template <typename T>
  class UnivariatePolynomial<T, -1>
  {
  public:
    using coefficient_type = T;

    //! @{
    //! @brief Constructors.
    inline UnivariatePolynomial() = default;

    inline explicit UnivariatePolynomial(int degree)
    {
      resize(degree);
    }
    //! @}

    //! @{
    //! @brief Return the polynomial coefficient for degree 'i'.
    inline auto operator[](int i) const -> const coefficient_type&
    {
      return _coeff[i];
    }

    inline auto operator[](int i) -> coefficient_type&
    {
      return _coeff[i];
    }
    //! @}

    //! @brief Return the polynomial degree.
    inline auto degree() const -> int
    {
      return static_cast<int>(_coeff.size()) - 1;
    }

    auto operator+(const UnivariatePolynomial& other) const
        -> UnivariatePolynomial
    {
      auto res = UnivariatePolynomial{std::max(degree(), other.degree())};
      for (auto i = 0u; i < _coeff.size(); ++i)
        res[i] += (*this)[i];
      for (auto j = 0u; j < other._coeff.size(); ++j)
        res[j] += other[j];
      return res;
    }

    auto operator+(const coefficient_type& other) const -> UnivariatePolynomial
    {
      auto res = *this;
      res[0] += other;
      return res;
    }

    auto operator-(const coefficient_type& other) const -> UnivariatePolynomial
    {
      return (*this) + (-other);
    }

    auto operator-(const UnivariatePolynomial& other) const
        -> UnivariatePolynomial
    {
      auto res = UnivariatePolynomial{std::max(degree(), other.degree())};

      for (auto i = 0u; i < _coeff.size(); ++i)
        res[i] += (*this)[i];

      for (auto j = 0u; j < other._coeff.size(); ++j)
        res[j] -= other[j];

      return res;
    }

    auto operator*(const UnivariatePolynomial& other) const
        -> UnivariatePolynomial
    {
      auto res = UnivariatePolynomial{degree() + other.degree()};
      for (auto i = 0u; i < _coeff.size(); ++i)
      {
        for (auto j = 0u; j < other._coeff.size(); ++j)
        {
          const auto& a = (*this)[i];
          const auto& b = other[j];
          res[i + j] += a * b;
        }
      }
      return res;
    }

    inline auto remove_leading_zeros() -> void
    {
      auto d = degree();
      while (std::abs(_coeff[d]) < std::numeric_limits<double>::epsilon())
        --d;
      resize(d);
    }

    inline auto resize(int degree) -> void
    {
      _coeff.resize(degree + 1);
    }

    //! @brief Euclidean division.
    auto operator/(const UnivariatePolynomial& other) const
        -> std::pair<UnivariatePolynomial, UnivariatePolynomial>
    {
      if (degree() < other.degree())
        return {*this, {}};

      auto a = *this;
      const auto& b = other;

      auto q = UnivariatePolynomial{degree() - other.degree()};
      q.resize(degree() - other.degree());

      auto qi = q;

      // Euclidean division.
      while (a.degree() >= b.degree())
      {
        qi[qi.degree()] = a[a.degree()] / b[b.degree()];

        a = a - b * qi;
        a.resize(a.degree() - 1);

        q = q + qi;

        qi.resize(qi.degree() - 1);
      }

      return {q, a};
    }

    inline auto operator/(const coefficient_type& other) const
        -> UnivariatePolynomial
    {
      auto res = *this;
      for (auto& c : res._coeff)
        c /= other;
      return res;
    }

    inline auto operator-() const -> UnivariatePolynomial
    {
      auto res = *this;
      for (auto& c : res._coeff)
        c = -c;
      return res;
    }

    //! @brief Horner method evaluation.
    template <typename U>
    auto operator()(U x) const -> decltype(T{} + U{})
    {
      if (x == U{})
        return _coeff[0];

      using result_type = decltype(T{} + U{});
      auto b = static_cast<result_type>(_coeff[degree()]);
      for (auto i = 1u; i < _coeff.size(); ++i)
        b = _coeff[degree() - i] + b * x;
      return b;
    }

    auto to_string() const -> std::string
    {
      auto str = std::string{};
      std::ostringstream oss;
      for (auto i = 0u; i < _coeff.size(); ++i)
      {
        oss << _coeff[degree() - i] << " X^" << (degree() - i);
        if (int(i) < degree())
          oss << " + ";
      }
      return oss.str();
    }

    friend inline auto operator<<(std::ostream& os,
                                  const UnivariatePolynomial& p)
        -> std::ostream&
    {
      os << p.to_string();
      return os;
    }

    std::vector<coefficient_type> _coeff;
  };


  //! @brief Univariate monomial class with runtime degree.
  class Monomial
  {
  public:
    Monomial() = default;

    template <typename T>
    auto pow(int e) const -> UnivariatePolynomial<T, -1>
    {
      auto P = UnivariatePolynomial<T, -1>{};
      P._coeff = std::vector<T>(exponent * e + 1, 0);
      P[exponent * e] = 1;
      return P;
    }

    template <typename T>
    inline auto to_polynomial() const -> UnivariatePolynomial<T, -1>
    {
      auto P = UnivariatePolynomial<T, -1>{};
      P._coeff = std::vector<T>(exponent + 1, T(0));
      P[exponent] = T(1);
      return P;
    }

    int exponent{1};
  };

  constexpr auto X = Monomial{};
  constexpr auto Z = Monomial{};


  template <typename T>
  auto operator+(const Monomial& a, const T& b)
  {
    auto res = UnivariatePolynomial<T, -1>{};
    res._coeff = std::vector<T>(a.exponent + 1, 0);
    res._coeff[a.exponent] = 1.;
    res._coeff[0] = b;
    return res;
  }

  template <typename T>
  auto operator-(const Monomial& a, const T& b)
  {
    return a + (-b);
  }

  template <typename T>
  auto operator*(const T& a, const Monomial& b)
  {
    auto res = UnivariatePolynomial<T, -1>{};
    res.resize(b.exponent);
    res._coeff[b.exponent] = a;
    return res;
  }

  template <typename T>
  auto operator*(const Monomial& a, const T& b)
  {
    return b * a;
  }

  template <typename T, int N>
  auto operator*(const T& a, const UnivariatePolynomial<T, N>& b)
  {
    auto res = b;
    for (auto i = 0u; i < res._coeff.size(); ++i)
      res[i] *= a;
    return res;
  }

  template <typename T, int N>
  auto operator*(const UnivariatePolynomial<T, N>& a, const T& b)
  {
    return b * a;
  }

  template <typename T, int N>
  auto operator*(const UnivariatePolynomial<T, N>& P, const Monomial& Q)
  {
    return P * Q.to_polynomial<T>();
  }

  template <typename T, int N>
  auto operator*(const Monomial& P, const UnivariatePolynomial<T, N>& Q)
  {
    return Q * P;
  }

  template <typename T, int N>
  auto operator/(const UnivariatePolynomial<T, N>& P, const Monomial& Q)
  {
    return P / Q.to_polynomial<T>();
  }

  //! @}

} /* namespace DO::Sara::Univariate */
