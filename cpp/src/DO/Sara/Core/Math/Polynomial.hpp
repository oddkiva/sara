#pragma once

#include <DO/Sara/Core/MultiArray.hpp>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>


namespace DO { namespace Sara {

  class Expression
  {
  public:
    Expression() = default;
    virtual ~Expression() = default;
  };


  class Variable : public Expression
  {
  public:
    Variable() = default;

    explicit Variable(const std::string& name)
      : _name(name)
    {
    }

    auto operator<(const Variable& other) const -> bool
    {
      return _name < other._name;
    }

    auto name() const -> const std::string&
    {
      return _name;
    }

  private:
      std::string _name;
  };


  class Monomial : public Expression
  {
  public:
    Monomial() = default;

    Monomial(const Variable& var)
      : exponents{{var, 1}}
    {
    }

    auto operator<(const Monomial& other) const -> bool
    {
      std::vector<Variable> vars;
      std::vector<Variable> other_vars;

      for (const auto& var: exponents)
        for (auto i = 0; i < var.second; ++i)
          vars.push_back(var.first);

      for (const auto& var: other.exponents)
        for (auto i = 0; i < var.second; ++i)
          other_vars.push_back(var.first);

      return std::lexicographical_compare(vars.begin(), vars.end(),
                                          other_vars.begin(), other_vars.end());
    }

    auto operator*(const Monomial& other) const -> Monomial
    {
      auto res = *this;
      // (x**2 y) * (x z)
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
      if (exponents.empty() || exponents.begin()->first.name().empty())
        return "1";

      auto str = std::string{};
      for (const auto& i : exponents)
      {
        if (i.second == 1)
          str += i.first.name() + " * ";
        else
          str += i.first.name() + "**" + std::to_string(i.second) + " * ";
      }

      str.pop_back();
      str.pop_back();

      return str;
    }
  private:
    std::map<Variable, int> exponents;
  };


  template <typename Coeff>
  class Polynomial : public Expression
  {
  public:
    Polynomial() = default;

    Polynomial operator+(const Polynomial& other) const
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

    Polynomial operator*(const Polynomial& other) const
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

    Polynomial t() const
    {
      auto res = *this;
      for (auto& i : res.coeffs)
        i.second.transposeInPlace();
      return res;
    }

    std::map<Monomial, Coeff> coeffs;
  };

  template <typename Matrix_>
  Polynomial<typename Matrix_::Scalar> trace(const Polynomial<Matrix_>& P)
  {
    using T = typename Matrix_::Scalar;
    auto res = Polynomial<T>{};
    for (auto& i : P.coeffs)
      res.coeffs[i.first] = i.second.trace();
    return res;
  }

  template <typename Scalar_, typename Matrix_>
  Polynomial<Matrix_> operator*(const Polynomial<Scalar_>& P, Polynomial<Matrix_>& Q)
  {
    auto res = Polynomial<Matrix_>{};

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

} /* namespace Sara */
} /* namespace DO */
