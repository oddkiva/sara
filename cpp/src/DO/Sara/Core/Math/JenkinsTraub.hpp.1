#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <complex>


namespace DO { namespace Sara {

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


} /* namespace Sara */
} /* namespace DO */
