#pragma once

#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>


namespace DO { namespace Sara {

  template <typename T>
  inline auto derivative(const UnivariatePolynomial<T>& P)
      -> UnivariatePolynomial<T>
  {
    auto Q = UnivariatePolynomial<T>{P.degree() - 1};
    for (auto i = 1; i <= P.degree(); ++i)
      Q[i - 1] = P[i] * i;
    return Q;
  }

  template <typename T>
  struct NewtonRaphson
  {
    NewtonRaphson(const UnivariatePolynomial<T>& poly)
      : p{poly}
      , p_prime{derivative(poly)}
    {
    }

    auto operator()(const T& x, int n = 1, double eps = 1e-2) const -> T
    {
      auto x0 = x;
      auto x1 = x;
      for (auto i = 0; i < n; ++i)
      {
        x0 = x1;
        x1 -= p(x1) / p_prime(x1);
        if (std::abs(x1 - x0) < eps)
          break;
      }
      return x1;
    }

    const UnivariatePolynomial<T>& p;
    UnivariatePolynomial<T> p_prime;
  };

} /* namespace Sara */
} /* namespace DO */
