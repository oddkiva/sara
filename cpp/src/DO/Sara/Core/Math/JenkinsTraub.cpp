#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <complex>
#include <ctime>
#include <memory>


namespace DO { namespace Sara {

  auto compute_moduli_lower_bound(const UnivariatePolynomial<double>& P)
      -> double
  {
    auto Q = P;
    Q[0] = -std::abs(Q[0] / Q[Q.degree()]);
    for (int i = 1; i <= Q.degree(); ++i)
      Q[i] = std::abs(Q[i] / Q[Q.degree()]);

    auto x = 1.;
    auto newton_raphson = NewtonRaphson<double>{Q};
    x = newton_raphson(x, 50);

    return x;
  }

  // Sigma is a real polynomial. So (s1, s2) is a pair of identical real numbers
  // or a conjugate complex pair.
  auto sigma_generic_formula(const std::complex<double>& s1)
      -> UnivariatePolynomial<double>
  {
    const auto a = std::real(s1);
    const auto b = std::imag(s1);
    return Z.pow<double>(2) - 2 * a * Z + (a * a + b * b);
  }

  // See formula (2.2) at page 547.
  // Don't use because overflow and underflow problems would occur (page 563).
  auto K1_generic_recurrence_formula(const UnivariatePolynomial<double>& K0,
                                     const UnivariatePolynomial<double>& P,
                                     const UnivariatePolynomial<double>& sigma,
                                     const std::complex<double>& s1)
      -> UnivariatePolynomial<double>
  {
    const auto s2 = std::conj(s1);

    Matrix2cd a, b, c;
    a <<  P(s1),  P(s2),  //
         K0(s1), K0(s2);

    b <<     K0(s1),     K0(s2),  //
         s1 * P(s1), s2 * P(s2);

    c << s1 * P(s1), s2 * P(s2),  //
              P(s1),      P(s2);

    const auto m = std::real(a.determinant() / c.determinant());
    const auto n = std::real(b.determinant() / c.determinant());

    return ((K0 + (m * Z + n)*P) / sigma).first;
  }

  // See formula (2.7) at page 548.
  auto
  sigma_formula_from_shift_polynomials(const UnivariatePolynomial<double>& K0,
                                       const UnivariatePolynomial<double>& K1,
                                       const UnivariatePolynomial<double>& K2,
                                       const std::complex<double>& s1)
      -> UnivariatePolynomial<double>
  {
    const auto s2 = std::conj(s1);
    const auto a2 = std::real(K1(s1) * K2(s2) - K1(s2) * K2(s1));
    const auto a1 = std::real(K0(s2) * K2(s1) - K0(s1) * K2(s2));
    const auto a0 = std::real(K0(s1) * K1(s2) - K0(s2) * K1(s1));

    // return (a2 * Z.pow(2) + a1 * Z + a0) / a2;
    return Z.pow<double>(2) + (a1 / a2) * Z + (a0 / a2);
  }

  // See formula at "Stage 1: no-shift process" at page 556.
  auto K1_no_shift_polynomial(const UnivariatePolynomial<double>& K0,
                              const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>
  {
    auto K1 = (K0 - (K0(0) / P(0)) * P) / Z;
    return K1.first;
  }


  // This is the scaled recurrence formula (page 563).
  auto K0_polynomial(const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>
  {
    return derivative(P) / P.degree();
  }

  // Return linear polynomial remainder of division of:
  // - P  / sigma
  // - K0 / sigma
  //
  // Used for stage 2 (fixed-shift process).
  // TODO: For stage 2, P(s1) and P(s2) are evaluated only once.
  //
  // Used for stage 3 (variable-shift process).
  // TODO: s1 and s2 are updated every time.
  auto calculate_remainders(const UnivariatePolynomial<double>& P,
                            const UnivariatePolynomial<double>& K0,
                            const UnivariatePolynomial<double>& sigma,
                            const std::complex<double>& s1)
      -> std::pair<UnivariatePolynomial<double>, UnivariatePolynomial<double>>
  {
    const auto s2 = std::conj(s1);

    // See stage 2 formula (9.7) (page 563).
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

    const auto P_remainder = b * (Z + u) + a;
    const auto K0_remainder = d * (Z + u) + c;

    return {P_remainder, K0_remainder};
  }


  //auto stage2(const UnivariatePolynomial<double>& K0_,
  //            const UnivariatePolynomial<double>& P,
  //            int L = 20) -> double
  //{
  //  const auto beta = compute_moduli_lower_bound(P);
  //  const auto s1 = beta; //* std::exp(i*rand());
  //  const auto s2 = std::conj(s1); //* std::exp(i*rand());
  //  const auto sigma = sigma_(s1);

  //  auto K0 = K0_;
  //  auto K1 = K1_stage2(P, K0, sigma, s1, std::conj(s1));
  //  auto K2 = K1_stage2(P, K1, sigma, s1, std::conj(s1));
  //  auto K3 = K1_stage2(P, K2, sigma, s1, std::conj(s1));
  //  auto K4 = K1_stage2(P, K3, sigma, s1, std::conj(s1));

  //  auto u = std::real(s1 + s2);
  //  auto t0 = s1 - (a0 - b0 * s2) / (c0 - d0 * s2);
  //  auto t1 = s1 - (a1 - b1 * s2) / (c1 - d1 * s2);
  //  auto t2 = s1 - (a2 - b2 * s2) / (c2 - d2 * s2);

  //  for (int i = 0; i < L; ++i)
  //  {
  //    K0 = K1;
  //    K1 = K2;
  //    K2 = K1_stage2(P, K1, sigma, s1, std::conj(s1));

  //    t0 = t1;
  //    t1 = t2;
  //    t2 = s1 - P(s1) / K2(s1);

  //    auto sigma_0 = sigma(K0, K1, K2, s1, s2);
  //    auto sigma_1 = sigma(K1, K2, K3, s1, s2);
  //    auto sigma_2 = sigma(K2, K3, K4, s1, s2);

  //    const auto v0 = sigma_0[0];
  //    const auto v1 = sigma_1[0];
  //    const auto v2 = sigma_2[0];

  //    // Convergence to rho_1?
  //    if (std::abs(t1 - t0) <= 0.5 * t0 && std::abs(t2 - t1) <= 0.5 * t1)
  //      return t2;

  //    // Convergence of sigma(z) to (z - rho_1) * (z - rho_2)?
  //    if (std::abs(v1 - v0) <= 0.5 * v0 && std::abs(v2 - v1) <= 0.5 * v1)
  //      return t2;
  //  }

  //  return std::numeric_limits<double>::infinity();
  //}

  //auto stage3(const UnivariatePolynomial<double>& K0_,
  //            const UnivariatePolynomial<double>& P,
  //            int L = 20) -> double
  //{
  //  const auto beta = compute_moduli_lower_bound(P);
  //  const auto s1 = beta; //* std::exp(i*rand());
  //  const auto s2 = std::conj(s1); //* std::exp(i*rand());
  //  const auto sigma = sigma_(s1);

  //  auto K0 = K0_;
  //  auto K1 = K1_stage2(P, K0, sigma, s1, std::conj(s1));
  //  auto K2 = K1_stage2(P, K1, sigma, s1, std::conj(s1));
  //  auto K3 = K1_stage2(P, K2, sigma, s1, std::conj(s1));
  //  auto K4 = K1_stage2(P, K3, sigma, s1, std::conj(s1));

  //  auto u = std::real(s1 + s2);
  //  auto t0 = s1 - (a0 - b0 * s2) / (c0 - d0 * s2);
  //  auto t1 = s1 - (a1 - b1 * s2) / (c1 - d1 * s2);
  //  auto t2 = s1 - (a2 - b2 * s2) / (c2 - d2 * s2);

  //  for (int i = 0; i < L; ++i)
  //  {
  //    K0 = K1;
  //    K1 = K2;
  //    K2 = K1_stage2(P, K1, sigma, s1, std::conj(s1));

  //    t0 = t1;
  //    t1 = t2;
  //    t2 = s1 - P(s1) / K2(s1);

  //    auto sigma_0 = sigma(K0, K1, K2, s1, s2);
  //    auto sigma_1 = sigma(K1, K2, K3, s1, s2);
  //    auto sigma_2 = sigma(K2, K3, K4, s1, s2);

  //    const auto v0 = sigma_0[0];
  //    const auto v1 = sigma_1[0];
  //    const auto v2 = sigma_2[0];

  //    // Convergence to rho_1?
  //    if (std::abs(t1 - t0) <= 0.5 * t0 && std::abs(t2 - t1) <= 0.5 * t1)
  //      return t2;

  //    // Convergence of sigma(z) to (z - rho_1) * (z - rho_2)?
  //    if (std::abs(v1 - v0) <= 0.5 * v0 && std::abs(v2 - v1) <= 0.5 * v1)
  //      return t2;
  //  }

  //  return std::numeric_limits<double>::infinity();
  //}
  //auto sigma_lambda(const UnivariatePolynomial<double>& K0,
  //                  const UnivariatePolynomial<double>& K1,
  //                  const UnivariatePolynomial<double>& K2,
  //                  const std::complex<double>& s1,
  //                  const std::complex<double>& s2)
  //    -> UnivariatePolynomial<double>
  //{
  //  // Use a, b, c, d, u, v.
  //  auto K0_s1 = c0 - d0 * s2, K0_s2 = c0 - d0 * s1;
  //  auto K1_s1 = c1 - d1 * s2, K1_s2 = c1 - d1 * s1;
  //  auto K2_s1 = c2 - d2 * s2, K2_s2 = c2 - d2 * s1;
  //  const auto det = K1_s1 * K2_s2 - K1_s2 * K2_s1;

  //  const auto m0 = K1_s1 * K2_s2 - K1_s2 * K2_s2;
  //  const auto m1 = K0_s2 * K2_s1 - K0_s1 * K2_s2;
  //  const auto m2 = K0_s1 * K1_s2 - K0_s2 * K1_s1;

  //  const auto sigma = (m0 * Z.pow<std::complex<double>>(2) + m1 * Z + m2) / det;

  //  return sigma;
  //}


  //sigma_lambda(



  //auto stage3(const UnivariatePolynomial<double>& K0,
  //            const UnivariatePolynomial<double>& sigma0,
  //            const UnivariatePolynomial<double>& P) -> void
  //{
  //}


  //auto sigma_(const std::complex<double>& s1) -> UnivariatePolynomial<double>
  //{
  //  auto res = UnivariatePolynomial<double>{};
  //  auto res_c = (Z - s1) * (Z - std::conj(s1));

  //  res._coeff.resize(res_c._coeff.size());
  //  for (auto i = 0u; i < res_c._coeff.size(); ++i)
  //    res[i] = std::real(res_c[i]);

  //  return res;
  //}

} /* namespace Sara */
} /* namespace DO */
