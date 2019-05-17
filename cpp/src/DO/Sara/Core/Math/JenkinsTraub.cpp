#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/JenkinsTraub.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>

#include <array>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>


// #define SHOW_DEBUG_LOG
#define LOG_DEBUG std::cout << "[" << __FUNCTION__ << ":" << __LINE__ << "] "


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
    Q[0] = -std::abs(Q[0]);
    for (int i = 1; i <= Q.degree(); ++i)
      Q[i] = std::abs(Q[i]);

    /*
     * Memento:
     *
     * If P(x) = 0 and x != 0, then with the triangle inequality:
     *   |a_0| <= |a_n| |x|^n + ... + |a_1| |x|
     *      0 <=  |a_n| |x|^n + ... + |a_1| |x| - |a_0|
     *      0 <=  Q(|x|)
     *
     * Q is increasing for x > 0 because Q' is positive.
     *
     * It suffices to find one root beta of Q because if
     * 0 == Q(beta) <= Q(|x|)
     *
     * Then beta <= |x| because Q is increasing.
     *
     * And we know beta > 0, because |a_0| is positive.
     * Otherwise 0 would already be a root of P.
     *
     * So for efficiency, we need to deflate P first.
     *
     */

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "Compute moduli lower bound" << std::endl;
    LOG_DEBUG << "P[X] = " << P << std::endl;
    LOG_DEBUG << "Q[X] = " << Q << std::endl;
#endif

    auto x = 1.;
    auto newton_raphson = NewtonRaphson<double>{Q};
    x = newton_raphson(x, 100, 1e-2);

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "Moduli lower bound = " << x << endl;
#endif

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
    // The two formula below are identical but the former might not very stable
    // numerically...
    //
    // auto K1 = ((K0 - (K0(0) / P(0)) * P) / Z).first;
    auto K1 = ((K0 - K0(0)) / Z).first - (K0(0) / P(0)) * (P / Z).first;

    return K1;
  }


  auto JenkinsTraub::determine_moduli_lower_bound() -> void
  {
    beta = compute_moduli_lower_bound(P);
  }

  auto JenkinsTraub::form_quadratic_divisor_sigma() -> void
  {
    constexpr auto i = std::complex<double>{0, 1};

    s1 = beta * std::exp(i * 49. * M_PI / 180.);
    s2 = std::conj(s1);

    sigma = Z.pow<double>(2) - 2 * std::real(s1) * Z + std::real(s1 * s2);

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "sigma[X] = " << sigma << endl;
    LOG_DEBUG << "s1 = " << s1 << endl;
    LOG_DEBUG << "s2 = " << s2 << endl;
#endif

    u = sigma[1];
    v = sigma[0];

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "  u = " << u << endl;
    LOG_DEBUG << "  v = " << v << endl;
#endif
  }

  auto JenkinsTraub::evaluate_polynomial_at_divisor_roots() -> void
  {
    P_s1 = P_r(s1);
    P_s2 = P_r(s2);

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "  P(s1) = " << "P(" << s1 << ") = " << P_s1 << endl;
    LOG_DEBUG << "  P(s2) = " << "P(" << s2 << ") = " << P_s2 << endl;
#endif
  }

  auto JenkinsTraub::evaluate_shift_polynomial_at_divisor_roots() -> void
  {
    K0_s1 = K0_r(s2);
    K0_s2 = K0_r(s1);

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "  K0(s1) = " << "K0(" << s1 << ") = " << K0_s1 << endl;
    LOG_DEBUG << "  K0(s2) = " << "K0(" << s2 << ") = " << K0_s2 << endl;
#endif
  }

  auto JenkinsTraub::calculate_coefficients_of_linear_remainders_of_P() -> void
  {
#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "P_r = " << P_r << endl;
#endif

    b = P_r[1];
    a = P_r[0] - b * u;

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "a = " << a << endl;
    LOG_DEBUG << "b = " << b << endl;
#endif
  }

  auto JenkinsTraub::calculate_coefficients_of_linear_remainders_of_K() -> void
  {
#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "K0_r = " << K0_r << endl;
#endif

    d = K0_r[1];
    c = K0_r[0] - d * u;

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "c = " << c << endl;
    LOG_DEBUG << "d = " << d << endl;
#endif
  }

  auto JenkinsTraub::calculate_next_shift_polynomial() -> void
  {
    const auto c0 = b * c - a * d;
    const auto c1 = (a * a + u * a * b + v * b * b) / c0;
    const auto c2 = (a * c + u * a * d + v * b * d) / c0;

    K1 = c1 * Q_K0 + (Z - c2) * Q_P + b;

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "K1 = " << K1 << std::endl;
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

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "b1 = " << b1 << endl;
    LOG_DEBUG << "b2 = " << b2 << endl;

    LOG_DEBUG << "a1 = " << a1 << endl;
    LOG_DEBUG << "a2 = " << a2 << endl;

    LOG_DEBUG << "c2 = " << c2 << endl;
    LOG_DEBUG << "c3 = " << c3 << endl;

    LOG_DEBUG << "c4 = " << c4 << endl;
    LOG_DEBUG << "c1 = " << c1 << endl;
    LOG_DEBUG << "v * b2 * a1 = " << v * b2 * a1 << endl;
#endif

    const auto delta_u = -(u * (c2 + c3) + v * (b1 * a1 + b2 * a2)) / c1;
    const auto delta_v = v * c4 / c1;

    sigma_shifted[0] = v + delta_v;
    sigma_shifted[1] = u + delta_u;
    sigma_shifted[2] = 1.0;

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "delta_u = " << delta_u << endl;
    LOG_DEBUG << "delta_v = " << delta_v << endl;
#endif
  }


  auto JenkinsTraub::stage1() -> void
  {
    LOG_DEBUG << "P[X] = " << P << endl;
    LOG_DEBUG << "[STAGE 1] " << endl;

    K0 = K0_polynomial(P);
    LOG_DEBUG << "[ITER] " << 0 << "  K[0] = " << K0 << endl;

    for (int i = 1; i < M; ++i)
    {
      K0 = K1_no_shift_polynomial(K0, P);
      LOG_DEBUG << "[ITER] " << i << "  K[" << i << "] = " << K0 << endl;
    }
  }

  auto JenkinsTraub::stage2() -> void
  {
    LOG_DEBUG << "[STAGE 2] " << endl;
    determine_moduli_lower_bound();

    // Stage 2 must be able to determine the convergence.
    while (cvg_type == NoConvergence)
    {
      // Choose roots randomly on the circle of radius beta.
      form_quadratic_divisor_sigma();

      std::tie(Q_P, P_r) = P / sigma;
      LOG_DEBUG << "  Q_P[X] = " << Q_P << endl;
      LOG_DEBUG << "  P_r[X] = " << P_r << endl;

      evaluate_polynomial_at_divisor_roots();
      calculate_coefficients_of_linear_remainders_of_P();

      sigma_shifted = sigma;
      auto t = std::array<std::complex<double>, 3>{{0., 0., 0}};
      auto v = std::array<double, 3>{0, 0, 0};

      // Determine convergence type.
      int i = M;
      for ( ; i < L; ++i)
      {
        std::tie(Q_K0, K0_r) = K0 / sigma;

        LOG_DEBUG << "[ITER] " << i << endl;
        LOG_DEBUG << "  K[" << i << "][X] = " << K0 << endl;
        LOG_DEBUG << "  Q_K[" << i << "][X] = " << Q_K0 << endl;
        LOG_DEBUG << "  K_r"<< i << "][X] = " << K0_r << endl;

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

        LOG_DEBUG << "  sigma[X] = " << sigma << endl;
        LOG_DEBUG << "  sigma_shifted[X] = " << sigma_shifted << endl;
        LOG_DEBUG << "  K[" << i << "] = " << K0 << endl;
        for (int k = 0; k < 3; ++k)
          LOG_DEBUG << "  t[" << k << "] = " << t[k] << endl;
        for (int k = 0; k < 3; ++k)
          LOG_DEBUG << "  v[" << k << "] = " << v[k] << endl;

        if (i < M + 3)
          continue;

        if (weak_convergence_predicate(t))
        {
          cvg_type = LinearFactor;
          s_i = std::real(t[2]);

          LOG_DEBUG << "Convergence to linear factor" << endl;
          LOG_DEBUG << "s = " << s_i << endl;

          break;
        }

        if (weak_convergence_predicate(v))
        {
          cvg_type = QuadraticFactor;
          v_i = sigma_shifted[0];

          LOG_DEBUG << "Convergence to quadratic factor" << endl;
          LOG_DEBUG << "v_i = " << v[2] << endl;

          break;
        }
      }

      L = i;

      // The while loop will keep going if cvg_type is NoConvergence.
    }
  }

  auto JenkinsTraub::stage3() -> void
  {
    LOG_DEBUG << "[STAGE 3] " << endl;

    LOG_DEBUG << "  s_i = " << s_i << endl;
    LOG_DEBUG << "  v_i = " << v_i << endl;

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

      LOG_DEBUG << "[ITER] " << i << endl;
      LOG_DEBUG << "  K[" << i << "] = " << K0 << endl;
      LOG_DEBUG << "  Sigma[" << i << "] = " << sigma_shifted << endl;

      if (cvg_type == LinearFactor)
      {
        s_i = s_i - P(s_i) / K1(s_i);
        z[0] = z[1];
        z[1] = z[2];
        z[2] = s_i;

        LOG_DEBUG << "  P_r(s_i) = "
                  << "P_r(" << s_i << ") = " << P_r(s_i) << std::endl;
        LOG_DEBUG << "  P(s_i) = "
                  << "P(" << s_i << ") = " << P(s_i) << std::endl;
        LOG_DEBUG << "  s[" << i << "] = " << s_i << endl;
      }

      if (cvg_type == QuadraticFactor)
      {
        v_i = sigma_shifted[2];
        z[0] = z[1];
        z[1] = z[2];
        z[2] = std::abs(s1) + std::abs(s2);

        LOG_DEBUG << "  v[" << i << "] = " << v_i << endl;
      }

      // Update K0.
      K0 = K1;

      if (std::isnan(z[2]))
      {
        LOG_DEBUG << "Stopping prematuraly at iteration " << i << endl;
        if (cvg_type == LinearFactor)
          s_i = z[1];
        // Finish
        break;
      }

      if (i < L + 3)
        continue;

      if (nikolajsen_root_convergence_predicate(z))
      {
        LOG_DEBUG << "Converged at iteration " << i << endl;
        break;
      }
    }

    if (cvg_type == LinearFactor)
      LOG_DEBUG << "L[X] = X - " << setprecision(12) << s_i << endl;

    if (cvg_type == QuadraticFactor)
    {
      LOG_DEBUG << "Sigma[X] = " << sigma_shifted << endl;
      LOG_DEBUG << "s1 = " << s1 << " P(s1) = " << P(s1) << endl;
      LOG_DEBUG << "s2 = " << s2 << " P(s2) = " << P(s2) << endl;
    }
  }

} /* namespace Sara */
} /* namespace DO */
