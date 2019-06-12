#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/Symbol.hpp>

#include <map>
#include <vector>


namespace DO::Sara {

  //! @{
  //! @brief These implementations assume commutativity of elements.
  class Monomial : public Expression
  {
  public:
    Monomial() = default;

    Monomial(const Symbol& var)
      : exponents{{var, 1}}
    {
    }

    auto operator<(const Monomial& other) const -> bool
    {
      std::vector<Symbol> vars;
      std::vector<Symbol> other_vars;

      for (const auto& var : exponents)
        for (auto i = 0; i < var.second; ++i)
          vars.push_back(var.first);

      for (const auto& var : other.exponents)
        for (auto i = 0; i < var.second; ++i)
          other_vars.push_back(var.first);

      return std::lexicographical_compare(vars.begin(), vars.end(),
                                          other_vars.begin(), other_vars.end());
    }

    auto operator==(const Monomial& other) const -> bool
    {
      return exponents == other.exponents;
    }

    auto operator*(const Monomial& other) const -> Monomial
    {
      // Boundary case 1.
      if (exponents == std::map<Symbol, int>{{one(), 1}})
        return other;

      // Boundary case 2.
      if (other.exponents == std::map<Symbol, int>{{one(), 1}})
        return *this;

      // E.g.: (x**2 y) * (x z)
      auto res = *this;

      for (const auto& i : other.exponents)
      {
        if (res.exponents.find(i.first) == res.exponents.end())
          res.exponents[i.first] = i.second;
        else
          res.exponents[i.first] += i.second;
      }

      return res;
    }

    auto pow(int exponent) const -> Monomial
    {
      auto res = *this;
      for (auto& i : res.exponents)
        i.second *= exponent;
      return res;
    }

    auto to_string() const -> std::string
    {
      auto str = std::string{};
      for (const auto& i : exponents)
      {
        if (i.second == 1)
          str += i.first.name + "*";
        else
          str += i.first.name + "^" + std::to_string(i.second) + "*";
      }

      str.pop_back();

      return str;
    }

    template <typename T>
    auto eval(const std::map<Symbol, T>& values) const
    {
      auto val = T(1);
      for (const auto x: exponents)
      {
        auto m = values.find(x.first);
        if (m == values.end())
          throw std::runtime_error{"Missing variable values!"};
        val *= std::pow(m->second, x.second);
      }
      return val;
    }

  private:
    std::map<Symbol, int> exponents;
  };


  template <typename Coeff>
  class Polynomial : public Expression
  {
  public:
    Polynomial() = default;

    auto operator+(const Polynomial& other) const
    {
      auto res = *this;

      for (const auto& i : other.coeffs)
      {
        if (res.coeffs.find(i.first) == res.coeffs.end())
          res.coeffs[i.first] = i.second;
        else
          res.coeffs[i.first] += i.second;
      }

      return res;
    }

    auto operator-(const Polynomial& other) const -> Polynomial
    {
      auto res = *this;

      for (const auto& i : other.coeffs)
      {
        if (res.coeffs.find(i.first) == res.coeffs.end())
          res.coeffs[i.first] = -i.second;
        else
          res.coeffs[i.first] -= i.second;
      }

      return res;
    }

    auto operator*(const Polynomial& other) const
    {
      auto res = Polynomial{};

      for (const auto& i : this->coeffs)
      {
        for (const auto& j : other.coeffs)
        {
          // Monomial calculation.
          const auto& Xi = i.first;
          const auto& Xj = j.first;
          const auto Xij = Xi * Xj;

          // Coefficient calculation.
          const auto& ai = i.second;
          const auto& aj = j.second;
          const auto aij = ai * aj;

          if (res.coeffs.find(Xij) == res.coeffs.end())
            res.coeffs[Xij] = aij;
          else
            res.coeffs[Xij] += aij;
        }
      }

      return res;
    }

    auto operator*(const Monomial& q) const
    {
      auto res = Polynomial{};
      for (const auto& c : coeffs)
      {
        const auto p = c.first * q;
        res.coeffs[p] = c.second;
      }
      return res;
    }

    auto operator*=(double scalar) -> Polynomial&
    {
      for (auto& c : coeffs)
        coeffs[c.first] *= scalar;
      return *this;
    }

    auto operator()(int i, int j) const
    {
      auto res = Polynomial<double>{};
      for (const auto& c : coeffs)
        res.coeffs[c.first] = c.second(i, j);

      return res;
    }

    auto operator==(const Polynomial& other) const
    {
      return coeffs == other.coeffs;
    }

    template <typename T>
    auto eval(const std::map<Symbol, T>& values) const
    {
      auto val = T(0);
      for (const auto x: coeffs)
        val += x.first.template eval<T>(values) * x.second;
      return val;
    }

    auto t() const
    {
      auto res = *this;
      for (auto& i : res.coeffs)
        i.second.transposeInPlace();
      return res;
    }

    auto to_string(double zero_thres = 1e-12) const
    {
      auto str = std::string{};
      for (const auto& m : coeffs)
      {
        if (std::abs(m.second) > zero_thres)
          str += std::to_string(m.second) + "*" + m.first.to_string() + " + ";
      }

      if (str.size() >= 2)
      {
        str.pop_back();
        str.pop_back();
      }

      if (str.empty())
        str = "0";

      return str;
    }

    std::map<Monomial, Coeff> coeffs;
  };
  //! @}


  //! @{
  //! @brief Multiply a coefficient by monomial and vice-versa.
  template <typename Coeff>
  auto operator*(const Coeff& c, const Monomial& m)
  {
    auto res = Polynomial<Coeff>{};
    res.coeffs[m] = c;
    return res;
  }

  template <typename Coeff>
  auto operator*(const Monomial& m, const Coeff& c)
  {
    return c * m;
  }
  //! @}


  //! @{
  //! @brief Multiply a coefficient by a polynomial and vice-versa.
  template <typename Coeff>
  auto operator*(const Coeff& a, const Polynomial<Coeff>& b)
  {
    auto res = b;
    res *= a;
    return res;
  }

  template <typename Coeff>
  auto operator*(const Polynomial<Coeff>& b, const Coeff& a)
  {
    return b * a;
  }
  //! @}


  //! @brief Multiply a monomial by a polynomial and vice-versa.
  template <typename Coeff>
  auto operator*(const Monomial& a, const Polynomial<Coeff>& b)
  {
    auto res = Polynomial<Coeff>{};
    for (const auto& bi : b.coeffs)
      res.coeffs[bi.first * a] = bi.second;
    return res;
  }


  //! @brief Multiply a monomial by a polynomial and vice-versa.
  template <typename T, int M, int N>
  auto operator*(const Polynomial<T>& P, const Polynomial<Matrix<T, M, N>>& Q)
  {
    auto res = Polynomial<Matrix<T, M, N>>{};

    for (const auto& i : P.coeffs)
    {
      for (const auto& j : Q.coeffs)
      {
        // Monomial calculation.
        const auto& Xi = i.first;
        const auto& Xj = j.first;
        const auto Xij = Xi * Xj;

        // Coefficient calculation.
        const auto& ai = i.second;
        const auto& aj = j.second;
        const auto aij = ai * aj;

        if (res.coeffs.find(Xij) == res.coeffs.end())
          res.coeffs[Xij] = aij;
        else
          res.coeffs[Xij] += aij;
      }
    }

    return res;
  }


  //! @{
  //! @brief Usual linear algebra operators.
  template <typename Matrix_>
  auto trace(const Polynomial<Matrix_>& P)
  {
    using T = typename Matrix_::Scalar;
    auto res = Polynomial<T>{};
    for (auto& i : P.coeffs)
      res.coeffs[i.first] = i.second.trace();
    return res;
  }

  template <typename T>
  auto det(const Polynomial<Matrix<T, 2, 2>>& P)
  {
    return P(0, 0) * P(1, 1) - P(0, 1) * P(1, 0);
  }

  template <typename T>
  auto det(const Polynomial<Matrix<T, 3, 3>>& P)
  {
    /*
     * 00 01 02
     * 10 11 12
     * 20 21 22
     */
    auto det0 = P(1, 1) * P(2, 2) - P(2, 1) * P(1, 2);
    auto det1 = P(1, 0) * P(2, 2) - P(2, 0) * P(1, 2);
    auto det2 = P(1, 0) * P(2, 1) - P(2, 0) * P(1, 1);

    return P(0, 0) * det0 - P(0, 1) * det1 + P(0, 2) * det2;
  }
  //! @}

} /* namespace DO::Sara */
