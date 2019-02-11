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

  auto K1_(const UnivariatePolynomial<double>& K0,
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

    std::cout << "a =\n" << a << std::endl;
    std::cout << "b =\n" << b << std::endl;
    std::cout << "c =\n" << c << std::endl;
    std::cout << "d =\n" << d << std::endl;
    std::cout << "da = " << da << std::endl;
    std::cout << "db = " << db << std::endl;
    std::cout << "dc = " << dc << std::endl;
    std::cout << "dd = " << dd << std::endl;

    return {};
  }


} /* namespace Sara */
} /* namespace DO */
