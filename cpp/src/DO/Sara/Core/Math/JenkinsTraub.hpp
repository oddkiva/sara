#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <complex>
#include <ctime>
#include <memory>
#include <random>


namespace DO::Sara {

  namespace detail {

    struct TargetPolynomial;
    struct QuadraticRealDivisor;
    struct ShiftPolynomial;

    struct TargetPolynomial
    {
      const UnivariatePolynomial<double>& P;

      TargetPolynomial(const UnivariatePolynomial<double>& P);

      auto evaluate_at_divisor_roots(const QuadraticRealDivisor& sigma) -> void;

      //! @{
      //! @brief P(s1) and P(s2).
      std::complex<double> P_s1;
      std::complex<double> P_s2;
      //! @}
    };

    struct QuadraticRealDivisor
    {
      const UnivariatePolynomial<double>& P;

      UnivariatePolynomial<double> sigma;
      std::complex<double> s1;
      std::complex<double> s2;

      //! @brief Lower bound of root moduli of polynomial P.
      double beta;

      //! @{
      //! @brief Random engine to initialize the roots s1 and s2.
      std::random_device rd;
      std::mt19937 gen;
      std::uniform_real_distribution<> dist{0, 94.0};
      //! @}

      // 1. Determine moduli lower bound $\beta$.
      QuadraticRealDivisor(const UnivariatePolynomial<double>& P);

      operator const UnivariatePolynomial<double>&() const;

      auto u() const -> const double&;

      auto v() const -> const double&;

      // 2. Form quadratic polynomial $\sigma(z)$.
      // (For stage 2)
      auto initialize_randomly() -> void;

      // 3.5 Calculate the new quadratic polynomial sigma (cf. formula 6.7).
      // cf. formula from Jenkins PhD dissertation.
      //
      // (for stage 2 and 3).
      auto update(const ShiftPolynomial& K) -> void;
    };

    struct ShiftPolynomial
    {
      const UnivariatePolynomial<double>& P;

      //! Stage 2: fixed-shift polynomial.
      //! Stage 3: variable-shift polynomial.
      UnivariatePolynomial<double> K0;
      UnivariatePolynomial<double> K1;

      //! @{
      //! @brief K0(s1) and K0(s2)
      std::complex<double> K0_s1;
      std::complex<double> K0_s2;
      //! @}

      //! Auxiliary variables resulting from the euclidean division by sigma.
      double a, b, c, d;  // coefficients of linear remainders.
      UnivariatePolynomial<double> Q_P, P_r;    // P / sigma
      UnivariatePolynomial<double> Q_K0, K0_r;  // K / sigma

      ShiftPolynomial(const UnivariatePolynomial<double>& P);

      // 3.2 Evaluate the fixed/variable-shift polynomial at divisor roots.
      auto evaluate_at_divisor_roots(const QuadraticRealDivisor& sigma) -> void;

      // 3.3 Calculate coefficients of linear remainders (cf. formula 9.7).
      auto calculate_coefficients_of_linear_remainders(
          const TargetPolynomial& P,          //
          const QuadraticRealDivisor& sigma)  //
          -> void;

      // 3.4 Calculate the next fixed/variable-shift polynomial (cf.
      // formula 9.8).
      auto update(const QuadraticRealDivisor& sigma) -> void;
    };

  } /* namespace detail */

  auto compute_moduli_lower_bound(const UnivariatePolynomial<double>& P)
      -> double;

  auto K0_polynomial(const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>;

  auto K1_no_shift_polynomial(const UnivariatePolynomial<double>& K0,
                              const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>;

  auto K1_stage1(const UnivariatePolynomial<double>& K0,
                 const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>;

  auto K1_stage2(const UnivariatePolynomial<double>& K0,
                 const UnivariatePolynomial<double>& P,
                 const UnivariatePolynomial<double>& sigma,
                 const std::complex<double>& s1, const std::complex<double>& s2)
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

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dist{0.0, 94.0};
    double beta;
    //! @}

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
    auto calculate_coefficients_of_linear_remainders() -> void;

    // 3.4 Calculate the next fixed/variable-shift polynomial (cf. formula 9.8).
    auto calculate_next_shift_polynomial() -> void;

    // 3.5 Calculate the new quadratic polynomial sigma (cf. formula 6.7).
    // cf. formula from Jenkins PhD dissertation.
    auto calculate_next_quadratic_divisor() -> void;

    auto check_convergence_linear_factor(
        const std::array<std::complex<double>, 3>& t) -> bool;

    auto check_convergence_quadratic_factor(const std::array<double, 3>& v)
        -> bool;

    //! Accentuate smaller zeros.
    auto stage1() -> void;

    //! Determine convergence type.
    auto stage2() -> void;

    auto stage3() -> void;
  };

} /* namespace DO::Sara */
