#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/JenkinsTraub.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>

#include <array>
#include <ctime>
#include <iostream>
#include <memory>


using namespace std;


namespace DO { namespace Sara {

  // Cauchy's method to calculate the polynomial lower bound.
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

  // This is the scaled recurrence formula (page 563).
  auto K0_polynomial(const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>
  {
    return derivative(P) / P.degree();
  }

  // See formula at "Stage 1: no-shift process" at page 556.
  auto K1_no_shift_polynomial(const UnivariatePolynomial<double>& K0,
                              const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>
  {
    //auto K1 = (K0 - (K0(0) / P(0)) * P) / Z;
    //return K1.first;
    //More efficient
    auto K1 = ((K0 - K0(0)) / Z).first - (K0(0) / P(0)) * (P / Z).first;
    K1 = K1 / K1[K1.degree()];
    return K1;
  }

  // Fixed-shift process.
  // 1. Determine moduli lower bound $\beta$.
  auto JenkinsTraub::determine_moduli_lower_bound() -> void
  {
    beta = compute_moduli_lower_bound(P);
  }

  // 2. Form polynomial sigma(z).
  auto JenkinsTraub::form_quadratic_divisor_sigma() -> void
  {
    constexpr auto i = std::complex<double>{0, 1};

    //const auto phase = dist(rd);
    s1 = beta * std::exp(i * 49. * M_PI / 180.);
    s2 = std::conj(s1);

    sigma = Z.pow<double>(2) - 2 * std::real(s1) * Z + std::real(s1 * s2);
    cout << "sigma[X] = " << sigma << endl;
    cout << "s1 = " << s1 << endl;
    cout << "s2 = " << s2 << endl;
  }

  // 3.1 Evaluate the polynomial at divisor roots.
  auto JenkinsTraub::evaluate_polynomial_at_divisor_roots() -> void
  {
    P_s1 = P(s1);
    P_s2 = P(s2);
    cout << "P(s1) = " << "P(" << s1 << ") = " << P_s1 << endl;
    cout << "P(s2) = " << "P(" << s2 << ") = " << P_s2 << endl;
  }

  // 3.2 Evaluate the fixed-shift polynomial at divisor roots.
  auto JenkinsTraub::evaluate_shift_polynomial_at_divisor_roots() -> void
  {
    K0_s1 = K0(s1);
    K0_s2 = K0(s2);
    cout << "K0(s1) = " << "K0(" << s1 << ") = " << K0_s1 << endl;
    cout << "K0(s2) = " << "K0(" << s2 << ") = " << K0_s2 << endl;
  }

  // 3.3 Calculate coefficient of linear remainders (cf. formula 9.7).
  auto JenkinsTraub::calculate_coefficients_of_linear_remainders() -> void
  {
    // See stage 2 formula (9.7) (page 563).
    Matrix4cd M;
    Vector4cd y;

    M <<
      1, -s2, 0,   0,
      0,   0, 1, -s2,
      1, -s1, 0,   0,
      0,   0, 1, -s1;

    y << P_s1, P_s2, K0_s1, K0_s2;
    Vector4cd x = M.colPivHouseholderQr().solve(y);

    a = std::real(x[0]);
    b = std::real(x[1]);
    c = std::real(x[2]);
    d = std::real(x[3]);

    u = -std::real(s1 + s2);
    v = std::real(s1 * s2);
  }

  // 3.4 Calculate the next fixed/variable-shift polynomial (cf. formula 9.8).
  auto JenkinsTraub::calculate_next_shift_polynomial() -> void
  {
    P_r = b * (Z + u) + a;
    K0_r = c * (Z + u) + d;

    Q_P = ((P - P_r) / sigma).first;
    Q_K0 = ((K0 - K0_r) / sigma).first;

    const auto c0 = b * c - a * d;
    const auto c1 = (a * a + u * a * b + v * b * b) / c0;
    const auto c2 = (a * c + u * a * d + v * b * d) / c0;

    K1 = c1 * Q_K0 + (Z - c2) * Q_P + b;
  }

  // 3.5 Calculate the new quadratic polynomial sigma (cf. formula 6.7).
  // cf. formula from Jenkins PhD dissertation.
  auto JenkinsTraub::calculate_next_quadratic_divisor() -> void
  {
    const auto b1 = -K0[0] / P[0];
    const auto b2 = K0[1] + b1 * P[1] / P[0];

    const auto a1 = b * c - a * d;
    const auto a2 = a * c + u * a * d + v * b * d;

    const auto c2 = b1 * a2;
    const auto c3 = b1 * b1 * (a * a + u * a * b + v * b * b);
    const auto c4 = v * b2 * a1 - c2 - c3;
    const auto c1 = c * c + u * c * d + v * d * d +
                    b1 * (a * c + u * b * c + v * b * d) - c4;

    const auto delta_u = -(u * (c2 + c3) + v * (b1 * a1 + b2 * a2)) / c1;
    const auto delta_v = v * c4 / c1;

    sigma[0] = v + delta_v;
    sigma[1] = u + delta_u;
    sigma[2] = 1.0;
  }

  auto JenkinsTraub::check_convergence_linear_factor(
      const std::array<std::complex<double>, 3>& t) -> bool
  {
    return std::abs(t[1] - t[0]) <= std::abs(t[0]) / 2 &&
           std::abs(t[2] - t[1]) <= std::abs(t[1]) / 2;
  }

  auto JenkinsTraub::check_convergence_quadratic_factor(
      const std::array<double, 3>& v) -> bool
  {
    return std::abs(v[1] - v[0]) <= std::abs(v[0]) / 2 &&
           std::abs(v[2] - v[1]) <= std::abs(v[1]) / 2;
  }

  //! Accentuate smaller zeros.
  auto JenkinsTraub::stage1() -> void
  {
    cout << "[STAGE 1] " << endl;
    cout << "[ITER] " << 0 << endl;
    K0 = K0_polynomial(P);
    cout << "  K[0] = " << K0 << endl;
    cout << "  K[0](0) = " << K0(0) << endl;

    for (int i = 1; i < M; ++i)
    {
      cout << "[ITER] " << i << endl;
      K0 = K1_no_shift_polynomial(K0, P);
      cout << "  K[" << i << "] = " << K0 << endl;
      cout << "  K[" << i << "](0) = " << K0(0) << endl;
    }
  }

  //! Determine convergence type.
  auto JenkinsTraub::stage2() -> void
  {
    cout << "[STAGE 2] " << endl;
    determine_moduli_lower_bound();

    // Stage 2 must be able to determine the convergence.
    while (cvg_type == NoConvergence)
    {
      // Choose roots randomly on the circle of radius beta.
      form_quadratic_divisor_sigma();

      // Do it only once.
      evaluate_polynomial_at_divisor_roots();

      auto t = std::array<std::complex<double>, 3>{{0., 0., s1}};
      auto v = std::array<double, 3>{0, 0, sigma[0]};

      // Determine convergence type.
      for (int i = M; i < L; ++i)
      {
        cout << "[ITER] " << i << endl;

        evaluate_shift_polynomial_at_divisor_roots();

        calculate_coefficients_of_linear_remainders();
        calculate_next_shift_polynomial();
        calculate_next_quadratic_divisor();

        t[0] = t[1];
        t[1] = t[2];
        t[2] = s1 - P_s1 / K0_s1;

        v[0] = v[1];
        v[1] = v[2];
        v[2] = sigma[0];

        K0 = K1;

        cout << "Sigma[X] = " << sigma << endl;
        cout << "  K[" << i << "] = " << K0 << endl;
        for (int k = 0; k < 3; ++k)
          cout << "  t[" << k << "] = " << t[k] << endl;
        for (int k = 0; k < 3; ++k)
          cout << "  v[" << k << "] = " << v[k] << endl;

        if (i < M + 3)
          continue;

        if (check_convergence_linear_factor(t))
        {
          cvg_type = LinearFactor;
          cout << "Convergence to linear factor" << endl;
          s_i = std::real(t[2]);
          cout << "s = " << s_i << endl;
          break;
        }

        if (check_convergence_quadratic_factor(v))
        {
          cvg_type = QuadraticFactor;
          cout << "Convergence to quadratic factor" << endl;
          cout << "s = " << v[2] << endl;
          break;
        }
      }

      // The while loop will keep going if cvg_type is NoConvergence.
    }
  }

  auto roots(UnivariatePolynomial<double>& P,
             std::complex<double>& s1,
             std::complex<double>& s2) -> void
  {
    const auto& a = P[2];
    const auto& b = P[1];
    const auto& c = P[0];
    const auto delta = std::complex<double>(b * b - 4 * a * c);
    s1 = (-b + std::sqrt(delta)) / (2 * a);
    s2 = (-b - std::sqrt(delta)) / (2 * a);
    std::cout << "  Sigma[X] = " << P << std::endl;
    std::cout << "  Sigma(" << s1 << ") = " << P(s1) << std::endl;
    std::cout << "  Sigma(" << s2 << ") = " << P(s2) << std::endl;
  }

  auto JenkinsTraub::stage3() -> void
  {
    cout << "[STAGE 3] " << endl;

    roots(sigma, s1, s2);
    evaluate_polynomial_at_divisor_roots();
    evaluate_shift_polynomial_at_divisor_roots();
    auto v_i = sigma[0];

    cout << "  s_i = " << s_i << endl;
    cout << "  v_i = " << v_i << endl;

    int i = L;

    // Determine convergence type.
    while (true)
    {
      ++i;

      evaluate_polynomial_at_divisor_roots();
      evaluate_shift_polynomial_at_divisor_roots();

      calculate_coefficients_of_linear_remainders();
      calculate_next_shift_polynomial();

      cout << "[ITER] " << i << endl;
      cout << "  K[" << i << "] = " << K0 << endl;
      cout << "  Sigma[" << i << "] = " << sigma << endl;

      if (cvg_type == LinearFactor)
      {
        s_i = s_i - P(s_i) / K1(s_i);
        // Check convergence.
        cout << "  s_i = " << s_i << endl;
      }

      if (cvg_type == QuadraticFactor)
      {
        v_i = sigma[2];
        // Check convergence.
        cout << "  v_i = " << v_i << endl;
      }

      // Update K0.
      K0 = K1;
      // Only update the next quadratic divisor.
      calculate_next_quadratic_divisor();
      {
        u = sigma[1];
        v = sigma[0];
      }

      if (i == L + 30)
        break;
    }
  }

} /* namespace Sara */
} /* namespace DO */
