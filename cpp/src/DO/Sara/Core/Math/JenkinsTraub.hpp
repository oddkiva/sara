#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
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

  auto K1_stage1(const UnivariatePolynomial<double>& K0,
                 const UnivariatePolynomial<double>& P,
                 const UnivariatePolynomial<double>& sigma)
      -> std::pair<UnivariatePolynomial<double>, UnivariatePolynomial<double>>
  {
    return (K0 - K0(0) - K0(0) / P(0) * (P - P(0))) / Z;
  }

  template <typename T>
  auto compute_root_radius(const UnivariatePolynomial<T>& P) -> T
  {
  }

} /* namespace Sara */
} /* namespace DO */
