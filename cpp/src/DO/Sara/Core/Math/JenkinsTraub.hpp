#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <complex>
#include <ctime>
#include <memory>


namespace DO { namespace Sara {

  //! @{
  //! @brief Tools needed for Jenkins-Traub algorithm.
  auto compute_root_moduli_lower_bound(const UnivariatePolynomial<double>& P)
      -> double;

  auto quadratic_roots(UnivariatePolynomial<double>& P)
      -> std::array<std::complex<double>, 2>;
  //! @}


  //! @{
  //! @brief Convenient data structure for Jenkins-Traub algorithm.
  //!
  //! In case the Jenkins-Traub algorithm has determined convergence to a linear
  //! factor, the algorithm returns this data structure.
  struct LinearFactor
  {
    auto initialize(double root) -> void
    {
      polynomial = Z - root;
    }

    auto root() -> const double&
    {
      return polynomial[0];
    }

    UnivariatePolynomial<double> polynomial;
  };

  //! In case the Jenkins-Traub algorithm has determined convergence to a
  //! quadratic factor, the algorithm returns this data structure.
  struct QuadraticFactor
  {
    auto initialize(const UnivariatePolynomial<double>& P,
                    double phase = 49. * M_PI / 180.,
                    bool recalculate_root_moduli_lower_bound = false) -> void;

    auto s1() const -> const std::complex<double>& { return roots[0]; }
    auto s2() const -> const std::complex<double>& { return roots[1]; }
    auto u() const -> const double& { return polynomial[1]; }
    auto v() const -> const double& { return polynomial[0]; }

    //! @brief The polynomial itself.
    UnivariatePolynomial<double> polynomial;

    //! @brief Memoized roots.
    std::array<std::complex<double>, 2> roots;

    //! @brief Memoized root moduli lower bound.
    double beta;
  };
  //! @}

  //! @brief Polynomial decomposition variables in Jenkins-Traub algorithm.
  struct AuxiliaryVariables
  {
    //! @{
    //! @brief Polynomial euclidean divisions.
    std::pair<UnivariatePolynomial<double>, UnivariatePolynomial<double>>
        P_div_sigma;
    std::pair<UnivariatePolynomial<double>, UnivariatePolynomial<double>>
        K0_div_sigma;
    //! @}

    //! @brief Coefficients related to the remainder polynomial of P.
    double a, b;

    //! @brief Coefficients related to the remainder polynomial of K0.
    double c, d;

    //! @brief Memoized values of P(s1) and P(s2).
    std::complex<double> P_s1, P_s2;

    //! @brief Memoized values of K0(s1) and K0(s2).
    std::complex<double> K0_s1, K0_s2;

    auto
    update_target_polynomial_aux_vars(const UnivariatePolynomial<double>& P,
                                      const QuadraticFactor& sigma) -> void;

    auto
    update_shift_polynomial_aux_vars(const UnivariatePolynomial<double>& K0,
                                     const QuadraticFactor& sigma) -> void;
  };
  //! @}


  //! @{
  //! @brief Calculate the shift polynomials by recurrence.
  auto initial_shift_polynomial(const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>;

  auto next_zero_shift_polynomial(const UnivariatePolynomial<double>& K0,
                                  const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>;

  auto next_linear_shift_polynomial(const UnivariatePolynomial<double>& K0,
                                    const UnivariatePolynomial<double>& P,
                                    double s_i, double P_si, double K0_si)
      -> UnivariatePolynomial<double>;

  auto next_quadratic_shift_polymomial(const QuadraticFactor& sigma,
                                       const AuxiliaryVariables& aux)
    -> UnivariatePolynomial<double>;
  //! @}


  //! @brief Calculate the next quadratic factor by recurrence when the
  //! Jenkins-Traub algorithm has determined convergence to a quadratic factor.
  auto next_quadratic_factor(QuadraticFactor& sigma,
                             const UnivariatePolynomial<double>& P,
                             const UnivariatePolynomial<double>& K0,
                             const AuxiliaryVariables& aux)
    -> QuadraticFactor;


  // Weak convergence test for stage 2 of the algorithm.
  template <typename T>
  auto weak_convergence_predicate(const std::array<T, 3>& t) -> bool
  {
    return std::abs(t[1] - t[0]) <= std::abs(t[0]) / 2 &&
           std::abs(t[2] - t[1]) <= std::abs(t[1]) / 2;
  }

  // Strong convergence test for stage 3 of the algorithm.
  //
  // Taken shamelessly from the implementation RPoly++ written by Chris Sweeney.
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


  //! @brief the pipeline for the root-finding algorithm.
  struct JenkinsTraub
  {
    JenkinsTraub(const UnivariatePolynomial<double>& P)
      : P{P}
    {
    }

    enum ConvergenceType : std::uint8_t
    {
      NoConvergence = 0,
      LinearFactor_ = 1,
      QuadraticFactor_ = 2
    };
    ConvergenceType cvg_type{NoConvergence};

    UnivariatePolynomial<double> P;

    UnivariatePolynomial<double> K0, K1;
    AuxiliaryVariables aux;

    LinearFactor linear_factor;

    int phase_try{0};
    QuadraticFactor sigma0, sigma1;

    int M{5};
    int L{20};
    int max_iter{10000000};

    double si;
    double vi;

    //! @brief Apply zero shift polynomial.
    auto stage1() -> void;

    //! @brief Apply fixed shift polynomial to determine the convergence type
    //! (to a linear factor or to a quadratic factor).
    auto stage2() -> void;

    //! @{
    //! @brief Apply variable shift polynomial to refine the factors very fast.
    auto stage3_linear_factor() -> ConvergenceType;
    auto stage3_quadratic_factor() -> ConvergenceType;
    //! @}
  };

} /* namespace Sara */
} /* namespace DO */
