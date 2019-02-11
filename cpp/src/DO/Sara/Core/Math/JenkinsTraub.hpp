#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <ctime>
#include <memory>


namespace DO { namespace Sara {

  UnivariatePolynomial<double> K0(const UnivariatePolynomial<double>& P)
  {
    auto K0 = UnivariatePolynomial<double>{P.degree()};
    const auto n = K0.degree();

    for (int i = 0; i <= n; ++i)
      K0[n - 1 - i] = ((n - 1 - i) * P[n - i]) / n;

    return K0;
  }

  auto K1(const UnivariatePolynomial<double>& K0,
          const UnivariatePolynomial<double>& P,
          const UnivariatePolynomial<double>& sigma,  //
          std::complex<double>& s1, std::complex<double>& s2)
      -> UnivariatePolynomial<double>
  {
    Matrix2cd a, b, c, d;
    a <<
      s1 * P(s1), s2 * P(s2),
           P(s1),      P(s2);
    c <<
      s1 * K0(s1), s2 * K0(s2),
           K0(s1),      K0(s2);

    b <<
           K0(s1),     K0(s2),
      s1 *  P(s1), s2 * P(s2);

    d <<
       P(s1),  P(s2),
      K0(s1), K0(s2);

    const auto da = a.determinant();
    const auto db = b.determinant();
    const auto dc = c.determinant();
    const auto dd = d.determinant();

    return {};
  }


} /* namespace Sara */
} /* namespace DO */
