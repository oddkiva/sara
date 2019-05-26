#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <complex>
#include <ctime>
#include <memory>


namespace DO::Sara {

  auto compute_moduli_lower_bound(const UnivariatePolynomial<double>& P)
      -> double;

  auto K0_polynomial(const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>;

  auto K1_no_shift_polynomial(const UnivariatePolynomial<double>& K0,
                              const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>;


  struct JenkinsTraub
  {
    enum ConvergenceType : std::uint8_t
    {
      NoConvergence = 0,
      LinearFactor = 1,
      QuadraticFactor = 2
    };

    ConvergenceType cvg_type{NoConvergence};

    //! @{
    //! @brief parameters.
    int M{5};
    int L{20};
    int max_iter{10000000};

    //! @brief Lower bound of root moduli of polynomial P.
    double beta;

    //! The polynomial of which we want to find the roots.
    UnivariatePolynomial<double> P;
    //! P(s1) and P(s2).
    std::complex<double> P_s1;
    std::complex<double> P_s2;

    //! Quadratic real polynomial divisor.
    UnivariatePolynomial<double> sigma;
    //! The roots of sigma.
    std::complex<double> s1;
    std::complex<double> s2;

    //! Stage 2: fixed-shift polynomial.
    //! Stage 3: variable-shift polynomial.
    UnivariatePolynomial<double> K0;
    UnivariatePolynomial<double> K1;
    //! K0(s1) and K0(s2)
    std::complex<double> K0_s1;
    std::complex<double> K0_s2;
    //! Auxiliary variables.
    double a, b, c, d;
    double u, v;
    UnivariatePolynomial<double> Q_P, P_r;
    UnivariatePolynomial<double> Q_K0, K0_r;

    //! Fixed-shift coefficients to update sigma.
    MatrixXcd K1_{3, 2};

    JenkinsTraub(const UnivariatePolynomial<double>& P)
      : P{P}
    {
    }

    // 1. Determine moduli lower bound $\beta$.
    auto determine_moduli_lower_bound() -> void;

    // 2. Form polynomial sigma(z).
    auto form_quadratic_divisor_sigma() -> void;

    // 3.1 Evaluate the polynomial at divisor roots.
    auto evaluate_polynomial_at_divisor_roots() -> void;

    // 3.2 Evaluate the fixed-shift polynomial at divisor roots.
    auto evaluate_shift_polynomial_at_divisor_roots() -> void;

    // 3.3 Calculate coefficient of linear remainders (cf. formula 9.7).
    auto calculate_coefficients_of_linear_remainders_of_P() -> void;
    auto calculate_coefficients_of_linear_remainders_of_K() -> void;

    // 3.4 Calculate the next fixed/variable-shift polynomial (cf. formula 9.8).
    auto calculate_next_shift_polynomial() -> void;

    // 3.5 Calculate the new quadratic polynomial sigma (cf. formula 6.7).
    // cf. formula from Jenkins PhD dissertation.
    auto calculate_next_shifted_quadratic_divisor() -> void;

    // Weak convergence test for stage 2 of the algorithm.
    template <typename T>
    auto weak_convergence_predicate(const std::array<T, 3>& t) -> bool
    {
      return std::abs(t[1] - t[0]) <= std::abs(t[0]) / 2 &&
             std::abs(t[2] - t[1]) <= std::abs(t[1]) / 2;
    }

    // Strong convergence test for stage 3 of the algorithm.
    //
    // Shamelessly taken from Rpoly++ written Chris Sweeney.
    //
    // Nikolajsen, Jorgen L. "New stopping criteria for iterative root finding."
    // Royal Society open science (2014)
    template <typename T>
    auto nikolajsen_root_convergence_predicate(const std::array<T, 3>& roots)
        -> bool
    {
      constexpr auto root_mag_tol = 1e-8;
      constexpr auto abs_tol = 1e-14;
      constexpr auto rel_tol = 1e-10;

      const auto e_i = std::abs(roots[2] - roots[1]);
      const auto e_i_minus_1 = std::abs(roots[1] - roots[0]);
      const auto mag_root = std::abs(roots[1]);
      if (e_i <= e_i_minus_1)
      {
        if (mag_root < root_mag_tol)
          return e_i < abs_tol;
        else
          return e_i / mag_root <= rel_tol;
      }

      return false;
    }

    //! Accentuate smaller zeros.
    auto stage1() -> void;

    //! @{
    //! @brief For stage 2 and 3.
    UnivariatePolynomial<double> sigma_shifted;
    double s_i;
    double v_i;
    //! @}

    //! Determine convergence type.
    auto stage2() -> void;

    //! Polish the linear factor or the quadratic factor.
    auto stage3() -> void;

    auto stage3_linear_shift() -> void;
    auto stage3_quadratic_shift() -> void;
  };

} /* namespace DO::Sara */
