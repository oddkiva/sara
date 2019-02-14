#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <complex>
#include <ctime>
#include <memory>


namespace DO { namespace Sara {

  auto K0_(const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>
  {
    const auto n = P.degree();
    auto K0 = UnivariatePolynomial<double>{n - 1};

    for (int i = 0; i < n; ++i)
      K0[n - 1 - i] = ((n - i) * P[n - i]) / n;

    return K0;
  }

  auto K1_stage1(const UnivariatePolynomial<double>& K0,
                 const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>
  {
    auto K1 = (K0 - (K0(0) / P(0)) * P) / Z;
    return K1.first;
  }

  auto K1_stage2(const UnivariatePolynomial<double>& K0,
                 const UnivariatePolynomial<double>& P,
                 const UnivariatePolynomial<double>& sigma,
                 const std::complex<double>& s1, const std::complex<double>& s2)
      -> UnivariatePolynomial<double>
  {
    Matrix4cd M;
    Vector4cd y;

    M << 1, -s2, 0,   0,
         0,   0, 1,  -1,
         1, -s1, 0,   0,
         0,   0, 1, -s1;

    y << P(s1), P(s2), K0(s1), K0(s2);
    Vector4cd x = M.colPivHouseholderQr().solve(y);

    const auto a = std::real(x[0]);
    const auto b = std::real(x[1]);
    const auto c = std::real(x[2]);
    const auto d = std::real(x[3]);

    const auto u = - std::real(s1 + s2);
    const auto v = - std::real(s1 * s2);

    auto Q_P = (P - b * (Z + u) + a) / sigma;
    auto Q_K = (K0 - d * (Z + u) + c) / sigma;

    auto m = (a*a + u*a*b + v*b*b) / (b*c - a*d);
    auto n = (a*c + u*a*d + v*b*d) / (b*c - a*d);

    return m * Q_K.first + ((Z - n)) * Q_P.first + b;
  }

  auto sigma_(const std::complex<double>& s1)
    -> UnivariatePolynomial<double>
  {
    auto res = UnivariatePolynomial<double>{};
    auto res_c = (Z - s1) * (Z - std::conj(s1));

    res._coeff.resize(res_c._coeff.size());
    for (auto i = 0u; i < res_c._coeff.size(); ++i)
      res[i] = std::real(res_c[i]);

    return res;
  }

  template <typename T>
  auto compute_moduli_lower_bound(const UnivariatePolynomial<T>& P) -> T
  {
    auto Q = P;
    Q[0] = -std::abs(Q[0] / Q[Q.degree()]);
    for (int i = 1; i <= Q.degree(); ++i)
      Q[i] = std::abs(Q[i] / Q[Q.degree()]);

    auto x = T(1);
    auto newton_raphson = NewtonRaphson<T>{Q};
    x = newton_raphson(x, 50);

    return x;
  }

} /* namespace Sara */
} /* namespace DO */
