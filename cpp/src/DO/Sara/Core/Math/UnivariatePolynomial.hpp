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
#include <array>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


namespace DO::Sara {

  //! @ingroup Core
  //! @defgroup Math Some mathematical tools
  //! @{

  //! @brief Base univariate polynomial class.
  template <typename Array>
  class UnivariatePolynomialBase
  {
  public:
    using array_type = Array;
    using coefficient_type = typename Array::value_type;

    //! @brief Default constructor.
    inline UnivariatePolynomialBase() noexcept = default;

    //! @brief Move constructor.
    inline UnivariatePolynomialBase(UnivariatePolynomialBase&& other) noexcept =
        default;

    //! @brief Copy constructor.
    inline UnivariatePolynomialBase(const UnivariatePolynomialBase& other)
      : _coeff{other._coeff}
    {
    }

    //! @brief Constructor from array.
    //! @{
    inline UnivariatePolynomialBase(array_type&& coeff) noexcept
      : _coeff{coeff}
    {
    }

    inline UnivariatePolynomialBase(const array_type& coeff)
      : _coeff{coeff}
    {
    }
    //! @}

    //! @brief Assignment operator.
    inline auto operator=(const UnivariatePolynomialBase& other)
        -> UnivariatePolynomialBase&
    {
      _coeff = other._coeff;
      return *this;
    }

    //! @brief Move assignment operator.
    inline auto operator=(UnivariatePolynomialBase&& other)
        -> UnivariatePolynomialBase&
    {
      _coeff = std::move(other._coeff);
      return *this;
    }

    //! @brief STL interface.
    //! @{
    inline auto begin()
    {
      return _coeff.begin();
    }

    inline auto end()
    {
      return _coeff.end();
    }

    inline auto begin() const
    {
      return _coeff.begin();
    }

    inline auto end() const
    {
      return _coeff.end();
    }
    //! @}

    //! @brief Return the degree of the polynomial.
    inline auto degree() const -> int
    {
      return static_cast<int>(_coeff.size()) - 1;
    }

    inline auto fill(coefficient_type value) -> void
    {
      std::fill(_coeff.begin(), _coeff.end(), value);
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

    inline auto operator/=(const coefficient_type& other)
        -> UnivariatePolynomialBase&
    {
      for (auto& c : _coeff)
        c /= other;
      return *this;
    }

    //! @{
    //! Comparison operator.
    inline auto operator==(const UnivariatePolynomialBase& other) const -> bool
    {
      return std::equal(_coeff.begin(), _coeff.end(), other._coeff.begin());
    }

    inline auto operator!=(const UnivariatePolynomialBase& other) const -> bool
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
                                  const UnivariatePolynomialBase& p)
        -> std::ostream&
    {
      os << p.to_string();
      return os;
    }
    //! @}

  protected:
    array_type _coeff;
  };


  template <typename T, int N = -1>
  class UnivariatePolynomial;

  //! @brief Univariate polynomial class with degree known at compile time.
  template <typename T, int N>
  class UnivariatePolynomial
    : public UnivariatePolynomialBase<std::array<T, N + 1>>
  {
    using base_type = UnivariatePolynomialBase<std::array<T, N + 1>>;
    using base_type::_coeff;

  public:
    using coefficient_type = typename base_type::coefficient_type;
    using base_type::degree;

    //! @{
    //! Constructors.
    inline UnivariatePolynomial() = default;

    inline UnivariatePolynomial(base_type&& other)
      : base_type{other}
    {
    }

    inline UnivariatePolynomial(const base_type& other)
      : base_type{other}
    {
    }

    inline explicit UnivariatePolynomial(const T* coeff)
    {
      std::copy(coeff, coeff + N + 1, _coeff.begin());
    }

    inline UnivariatePolynomial(std::initializer_list<T> list)
    {
      std::copy(list.begin(), list.end(), _coeff.begin());
    }
    //! @}
  };

  //! @brief Univariate polynomial class with degree known at runtime.
  template <typename T>
  class UnivariatePolynomial<T, -1>
    : public UnivariatePolynomialBase<std::vector<T>>
  {
    using base_type = UnivariatePolynomialBase<std::vector<T>>;
    using base_type::_coeff;

  public:
    using base_type::degree;
    using array_type = typename base_type::array_type;
    using coefficient_type = typename base_type::coefficient_type;

    //! @{
    //! @brief Constructors.
    inline UnivariatePolynomial() = default;

    inline explicit UnivariatePolynomial(int degree)
    {
      resize(degree);
    }

    inline explicit UnivariatePolynomial(const array_type& coeff)
      : base_type{coeff}
    {
    }
    //! @}

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
  };


  //! @brief Univariate monomial class with runtime degree.
  class UnivariateMonomial
  {
  public:
    UnivariateMonomial() = default;

    template <typename T>
    auto pow(int e) const -> UnivariatePolynomial<T, -1>
    {
      auto P = UnivariatePolynomial<T, -1>{exponent * e};
      P.fill(0);
      P[exponent * e] = 1;
      return P;
    }

    template <typename T>
    inline auto to_polynomial() const -> UnivariatePolynomial<T, -1>
    {
      auto P = UnivariatePolynomial<T, -1>{exponent};
      P.fill(0);
      P[exponent] = T(1);
      return P;
    }

    int exponent{1};
  };

  constexpr auto X = UnivariateMonomial{};
  constexpr auto Z = UnivariateMonomial{};


  template <typename T>
  auto operator+(const UnivariateMonomial& a, const T& b)
  {
    auto res = UnivariatePolynomial<T, -1>(a.exponent);
    if (a.exponent != 0)
    {
      res[a.exponent] = 1;
      res[0] = b;
    }
    else
      res[a.exponent] = 1 + b;
    return res;
  }

  template <typename T>
  auto operator-(const UnivariateMonomial& a, const T& b)
  {
    return a + (-b);
  }

  template <typename T>
  auto operator*(const T& a, const UnivariateMonomial& b)
  {
    auto res = UnivariatePolynomial<T, -1>{};
    res.resize(b.exponent);
    res[b.exponent] = a;
    return res;
  }

  template <typename T>
  auto operator*(const UnivariateMonomial& a, const T& b)
  {
    return b * a;
  }

  template <typename T, int N>
  auto operator*(const T& a, const UnivariatePolynomial<T, N>& b)
  {
    auto res = b;
    for (auto i = 0; i <= res.degree(); ++i)
      res[i] *= a;
    return res;
  }

  template <typename T, int N>
  auto operator*(const UnivariatePolynomial<T, N>& a, const T& b)
  {
    return b * a;
  }

  template <typename T, int N>
  auto operator*(const UnivariatePolynomial<T, N>& P,
                 const UnivariateMonomial& Q)
  {
    return P * Q.to_polynomial<T>();
  }

  template <typename T, int N>
  auto operator*(const UnivariateMonomial& P,
                 const UnivariatePolynomial<T, N>& Q)
  {
    return Q * P;
  }

  template <typename T, int N>
  auto operator/(const UnivariatePolynomial<T, N>& P,
                 const UnivariateMonomial& Q)
  {
    return P / Q.to_polynomial<T>();
  }

  //! @}

}  // namespace DO::Sara
