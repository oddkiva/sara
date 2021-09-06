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

  template <typename T, int N>
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
      std::copy(list.begin(), list.end(), _coeff);
    }

    inline UnivariatePolynomial(const UnivariatePolynomial& P)
    {
      copy(P);
    }
    //! @}

    //! Assign a new polynomial to the polynomial object.
    UnivariatePolynomial& operator=(const UnivariatePolynomial& P)
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
    inline T& operator[](int degree)
    {
      return _coeff[degree];
    }

    inline const T& operator[](int degree) const
    {
      return _coeff[degree];
    }
    //! @}

    //! @brief Evaluate polynomial at point 'x' using Horner method evaluation.
    template <typename U>
    inline auto operator()(const U& x) const
        -> decltype(coefficient_type{} + U{})
    {
      if (x == U{})
        return _coeff[0];

      using result_type = decltype(coefficient_type{} + T{});
      auto b = result_type(_coeff[degree()]);
      for (auto i = 1; i < _coeff.size(); ++i)
        b = _coeff[degree() - i] + b * x;
      return b;
    }

    //! @{
    //! Comparison operator.
    inline bool operator==(const UnivariatePolynomial& other) const
    {
      for (int i = 0; i <= N; ++i)
        if (_coeff[i] != other._coeff[i])
          return false;
      return true;
    }

    inline bool operator!=(const UnivariatePolynomial& other) const
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
        if (signum(P[i]) >= 0)
          oss << "+";
        else
          oss << "-";
        oss << std::abs(P[i]);
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
    inline void copy(const UnivariatePolynomial& other)
    {
      std::copy(other._coeff, other._coeff + N + 1, _coeff);
    }

  private:
    std::array<T, N + 1> _coeff;
  };

  //! @brief Univariate polynomial class with degree known at runtime.
  template <typename Coeff>
  class UnivariatePolynomial<Coeff, -1>
  {
  public:
    using coefficient_type = Coeff;

    inline UnivariatePolynomial() = default;

    inline explicit UnivariatePolynomial(int degree)
      : _coeff(degree + 1, 0)
    {
    }

    const coefficient_type& operator[](int i) const
    {
      return _coeff[i];
    }

    coefficient_type& operator[](int i)
    {
      return _coeff[i];
    }

    int degree() const
    {
      return int(_coeff.size()) - 1;
    }

    UnivariatePolynomial operator+(const UnivariatePolynomial& other) const
    {
      auto res = UnivariatePolynomial{std::max(this->degree(), other.degree())};
      for (auto i = 0u; i < this->_coeff.size(); ++i)
        res[i] += (*this)[i];
      for (auto j = 0u; j < other._coeff.size(); ++j)
        res[j] += other[j];
      return res;
    }

    UnivariatePolynomial operator+(const coefficient_type& other) const
    {
      auto res = *this;
      res._coeff[0] += other;
      return res;
    }

    UnivariatePolynomial operator-(const coefficient_type& other) const
    {
      return (*this) + (-other);
    }

    UnivariatePolynomial operator-(const UnivariatePolynomial& other) const
    {
      auto res = UnivariatePolynomial{std::max(this->degree(), other.degree())};

      for (auto i = 0u; i < this->_coeff.size(); ++i)
        res[i] += (*this)[i];

      for (auto j = 0u; j < other._coeff.size(); ++j)
        res[j] -= other[j];

      return res;
    }

    UnivariatePolynomial operator*(const UnivariatePolynomial& other) const
    {
      auto res = UnivariatePolynomial{degree() + other.degree()};
      for (auto i = 0u; i < this->_coeff.size(); ++i)
        for (auto j = 0u; j < other._coeff.size(); ++j)
        {
          const auto& a = (*this)[i];
          const auto& b = other[j];
          res[i + j] += a * b;
        }
      return res;
    }

    auto remove_leading_zeros()
    {
      auto d = degree();
      while (std::abs(_coeff[d]) < std::numeric_limits<double>::epsilon())
        --d;
      resize(d);
    }

    auto resize(int degree) -> void
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
      q._coeff =
          std::vector<coefficient_type>(degree() - other.degree() + 1, 0);

      auto qi = q;

      // Euclidean division.
      while (a.degree() >= b.degree())
      {
        qi._coeff[qi.degree()] = a[a.degree()] / b[b.degree()];

        a = a - b * qi;
        a.resize(a.degree() - 1);

        q = q + qi;

        qi.resize(qi.degree() - 1);
      }

      return {q, a};
    }

    auto operator/(const coefficient_type& other) const -> UnivariatePolynomial
    {
      auto res = *this;
      for (auto& c : res._coeff)
        c /= other;
      return res;
    }

    UnivariatePolynomial operator-() const
    {
      auto res = *this;
      for (auto& c : res._coeff)
        c = -c;
      return res;
    }

    //! @brief Horner method evaluation.
    template <typename T>
    auto operator()(const T& x0) const -> decltype(coefficient_type{} + T{})
    {
      if (x0 == T(0))
        return _coeff[0];

      using result_type = decltype(coefficient_type{} + T{});
      auto b = result_type(_coeff[degree()]);
      for (auto i = 1u; i < _coeff.size(); ++i)
        b = _coeff[degree() - i] + b * x0;
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

    friend std::ostream& operator<<(std::ostream& os,
                                    const UnivariatePolynomial& p)
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
