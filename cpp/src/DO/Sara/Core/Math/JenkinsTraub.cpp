#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/JenkinsTraub.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>

#include <array>
#include <ctime>
#include <iostream>
#include <memory>


using namespace std;


namespace DO { namespace Sara {

  auto quadratic_roots(UnivariatePolynomial<double>& P,
                       std::complex<double>& s1, std::complex<double>& s2)
      -> void
  {
    const auto& a = P[2];
    const auto& b = P[1];
    const auto& c = P[0];
    const auto delta = std::complex<double>(b * b - 4 * a * c);
    s1 = (-b + std::sqrt(delta)) / (2 * a);
    s2 = (-b - std::sqrt(delta)) / (2 * a);
  }


  auto compute_moduli_lower_bound(const UnivariatePolynomial<double>& P)
      -> double
  {
    auto Q = P;
    Q[0] = -std::abs(Q[0] / Q[Q.degree()]);
    for (int i = 1; i <= Q.degree(); ++i)
      Q[i] = std::abs(Q[i] / Q[Q.degree()]);

    auto x = 1.;
    auto newton_raphson = NewtonRaphson<double>{Q};
    x = newton_raphson(x, 100, 1e-3);

    return x;
  }


  auto K0_polynomial(const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>
  {
    return derivative(P) / P.degree();
  }

  auto K1_no_shift_polynomial(const UnivariatePolynomial<double>& K0,
                              const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>
  {
#ifdef DEBUG_JT
    std::cout << "[" << __FUNCTION__ << ":" << __LINE__ << "] "
              << "P(0) = " << P(0) << std::endl;
#endif

    // The two formula below are identical but the former might not very stable
    // numerically...
    //auto K1 = ((K0 - (K0(0) / P(0)) * P) / Z).first;
    auto K1 = ((K0 - K0(0)) / Z).first - (K0(0) / P(0)) * (P / Z).first;
    return K1;
  }


  auto JenkinsTraub::determine_moduli_lower_bound() -> void
  {
    beta = compute_moduli_lower_bound(P);

#ifdef DEBUG_JT
    std::cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] "
              << "root radius of sigma = " << beta << std::endl;
#endif
  }

  auto JenkinsTraub::form_quadratic_divisor_sigma() -> void
  {
    constexpr auto i = std::complex<double>{0, 1};

    //const auto phase = dist(rd);
    s1 = beta * std::exp(i * 49. * M_PI / 180.);
    s2 = std::conj(s1);

    sigma = Z.pow<double>(2) - 2 * std::real(s1) * Z + std::real(s1 * s2);

#ifdef DEBUG_JT
    std::cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] "
              << "sigma[X] = " << sigma << endl;
    std::cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] "
              << "s1 = " << s1 << endl;
    std::cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] "
              << "s2 = " << s2 << endl;
#endif

    u = sigma[1];
    v = sigma[0];

    std::cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] "
              << "  u = " << u << endl;
    std::cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] "
              << "  v = " << v << endl;
  }

  auto JenkinsTraub::evaluate_polynomial_at_divisor_roots() -> void
  {
    P_s1 = P_r(s1);
    P_s2 = P_r(s2);
#ifdef DEBUG_JT
    cout << "  P(s1) = " << "P(" << s1 << ") = " << P_s1 << endl;
    cout << "  P(s2) = " << "P(" << s2 << ") = " << P_s2 << endl;
#endif
  }

  auto JenkinsTraub::evaluate_shift_polynomial_at_divisor_roots() -> void
  {
    K0_s1 = K0_r(s2);
    K0_s2 = K0_r(s1);
#ifdef DEBUG_JT
    cout << "  K0(s1) = " << "K0(" << s1 << ") = " << K0_s1 << endl;
    cout << "  K0(s2) = " << "K0(" << s2 << ") = " << K0_s2 << endl;
#endif
  }

  auto JenkinsTraub::calculate_coefficients_of_linear_remainders_of_P() -> void
  {
    b = P_r[1];
    a = P_r[0] - b * u;

#ifdef DEBUG_JT
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "a = " << a << endl;
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "b = " << b << endl;
#endif
  }

  auto JenkinsTraub::calculate_coefficients_of_linear_remainders_of_K() -> void
  {
    d = K0_r[1];
    c = K0_r[0] - d * u;

#ifdef DEBUG_JT
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "c = " << c << endl;
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "d = " << d << endl;
#endif
  }

  auto JenkinsTraub::calculate_next_shift_polynomial() -> void
  {
    const auto c0 = b * c - a * d;
    const auto c1 = (a * a + u * a * b + v * b * b) / c0;
    const auto c2 = (a * c + u * a * d + v * b * d) / c0;

    K1 = c1 * Q_K0 + (Z - c2) * Q_P + b;
    K1 = K1 / K1[K1.degree()];

#ifdef DEBUG_JT
    std::cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] "
              << "K1 = " << K1 << std::endl;
#endif
  }

  auto JenkinsTraub::calculate_next_shifted_quadratic_divisor() -> void
  {
    const auto b1 = -K0[0] / P[0];
    const auto b2 = -(K0[1] + b1 * P[1]) / P[0];

    const auto a1 = b * c - a * d;
    const auto a2 = a * c + u * a * d + v * b * d;

    const auto c2 = b1 * a2;
    const auto c3 = b1 * b1 * (a * a + u * a * b + v * b * b);

    const auto c4 = v * b2 * a1 - c2 - c3;
    const auto c1 = c * c + u * c * d + v * d * d +
                    b1 * (a * c + u * b * c + v * b * d) - c4;

#ifdef DEBUG_JT
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "b1 = " << b1 << endl;
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "b2 = " << b2 << endl;

    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "a1 = " << a1 << endl;
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "a2 = " << a2 << endl;

    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "c2 = " << c2 << endl;
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "c3 = " << c3 << endl;

    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "c4 = " << c4 << endl;
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "c1 = " << c1 << endl;
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "v * b2 * a1 = " << v * b2 * a1 << endl;
#endif

    const auto delta_u = -(u * (c2 + c3) + v * (b1 * a1 + b2 * a2)) / c1;
    const auto delta_v = v * c4 / c1;

    sigma_shifted[0] = v + delta_v;
    sigma_shifted[1] = u + delta_u;
    sigma_shifted[2] = 1.0;

#ifdef DEBUG_JT
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "delta_u = " << delta_u << endl;
    cout << "  [" << __FUNCTION__ << ":" << __LINE__ << "] " << "delta_v = " << delta_v << endl;
#endif
  }


  auto JenkinsTraub::stage1() -> void
  {
    cout << "[STAGE 1] " << endl;
    cout << "[ITER] " << 0 << endl;
    K0 = K0_polynomial(P);
    cout << "  K[0] = " << K0 << endl;

    for (int i = 1; i < M; ++i)
    {
      cout << "[ITER] " << i << endl;
      K0 = K1_no_shift_polynomial(K0, P);
      cout << "  K[" << i << "] = " << K0 << endl;
    }
  }

  auto JenkinsTraub::stage2() -> void
  {
    cout << "[STAGE 2] " << endl;
    determine_moduli_lower_bound();

    // Stage 2 must be able to determine the convergence.
    while (cvg_type == NoConvergence)
    {
      // Choose roots randomly on the circle of radius beta.
      form_quadratic_divisor_sigma();

      std::tie(Q_P, P_r) = P / sigma;
      cout << "  Q_P[X] = " << Q_P << endl;
      cout << "  P_r[X] = " << P_r << endl;
      evaluate_polynomial_at_divisor_roots();
      calculate_coefficients_of_linear_remainders_of_P();

      sigma_shifted = sigma;
      auto t = std::array<std::complex<double>, 3>{{0., 0., 0}};
      auto v = std::array<double, 3>{0, 0, 0};

      K0 = K0 / K0[K0.degree()];

      // Determine convergence type.
      int i = M;
      for ( ; i < L; ++i)
      {
        cout << "[ITER] " << i << endl;

        std::tie(Q_K0, K0_r) = K0 / sigma;
        cout << "  Ki[X] = " << K0 << endl;
        cout << "  Q_Ki[X] = " << Q_K0 << endl;
        cout << "  Ki_r[X] = " << K0_r << endl;
        evaluate_shift_polynomial_at_divisor_roots();

        calculate_coefficients_of_linear_remainders_of_K();

        t[0] = t[1];
        t[1] = t[2];
        t[2] = s1 - P_s1 / K0_s1;

        v[0] = v[1];
        v[1] = v[2];
        v[2] = sigma_shifted[0];

        calculate_next_shifted_quadratic_divisor();
        calculate_next_shift_polynomial();

        K0 = K1;

        //cout << "  sigma[X] = " << sigma << endl;
        cout << "  sigma_shifted[X] = " << sigma_shifted << endl;
        cout << "  K[" << i << "] = " << K0 << endl;
        for (int k = 0; k < 3; ++k)
          cout << "  t[" << k << "] = " << t[k] << endl;
        for (int k = 0; k < 3; ++k)
          cout << "  v[" << k << "] = " << v[k] << endl;

        if (i < M + 3)
          continue;

        if (weak_convergence_predicate(t))
        {
          cvg_type = LinearFactor;
          cout << "Convergence to linear factor" << endl;
          s_i = std::real(t[2]);
          cout << "s = " << s_i << endl;
          break;
        }

        if (weak_convergence_predicate(v))
        {
          cvg_type = QuadraticFactor;
          cout << "Convergence to quadratic factor" << endl;
          v_i = sigma_shifted[0];
          cout << "v_i = " << v[2] << endl;
          break;
        }
      }

      L = i;

      // The while loop will keep going if cvg_type is NoConvergence.
    }
  }

  auto JenkinsTraub::stage3() -> void
  {
    cout << "[STAGE 3] " << endl;

    cout << "  s_i = " << s_i << endl;
    cout << "  v_i = " << v_i << endl;

    int i = L;

    auto z = std::array<double, 3>{0., 0., 0.};

    // Determine convergence type.
    while (i < L + max_iter)
    {
      ++i;

      quadratic_roots(sigma_shifted, s1, s2);
      {
        u = sigma_shifted[1];
        v = sigma_shifted[0];
      }

      std::tie(Q_P, P_r) = P / sigma_shifted;
      evaluate_polynomial_at_divisor_roots();
      calculate_coefficients_of_linear_remainders_of_P();

      std::tie(Q_K0, K0_r) = K0 / sigma_shifted;
      evaluate_shift_polynomial_at_divisor_roots();
      calculate_coefficients_of_linear_remainders_of_K();

      calculate_next_shift_polynomial();
      calculate_next_shifted_quadratic_divisor();

      cout << "[ITER] " << i << endl;
      cout << "  K[" << i << "] = " << K0 << endl;
      cout << "  Sigma[" << i << "] = " << sigma_shifted << endl;

      if (cvg_type == LinearFactor)
      {
        s_i = s_i - P(s_i) / K1(s_i);
        if (i % 100000 == 0)
          cout << "  s[" << i << "] = " << s_i << endl;
        z[0] = z[1];
        z[1] = z[2];
        z[2] = s_i;
      }

      if (cvg_type == QuadraticFactor)
      {
        v_i = sigma_shifted[2];
        if (i % 100000 == 0)
          cout << "  v[" << i << "] = " << v_i << endl;
        z[0] = z[1];
        z[1] = z[2];
        z[2] = std::abs(s1);
      }

      // Update K0.
      K0 = K1;

      if (i < L + 3)
        continue;

      if (nikolajsen_root_convergence_predicate(z))
      {
        cout << "Converged at iteration " << i << endl;
        break;
      }
    }

    if (cvg_type == LinearFactor)
      cout << "L[X] = X - " << s_i << endl;

    if (cvg_type == QuadraticFactor)
    {
      cout << "Sigma[X] = " << sigma_shifted << endl;
      cout << "s1 = " << s1 << endl;
      cout << "s2 = " << s2 << endl;
    }
  }

} /* namespace Sara */
} /* namespace DO */
