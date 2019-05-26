#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>


namespace DO::Sara {

  struct LinearFactor
  {
    auto initialize(double shift) -> void
    {
      sigma = Z - shift;
    }

    auto root() -> const double&
    {
      return polynomial[0];
    }

    const UnivariatePolynomial<double> polynomial;
  };


  struct QuadraticFactor
  {
    auto initialize(const Univariate<double>& P,
                    double root_moduli_lower_bound,
                    double phase = 49. * M_PI / 180.) -> void
    {
      constexpr auto i = std::complex<double>{0, 1};

      const auto beta = compute_root_moduli_lower_bound(P);

      auto& [s1, s2] = roots;
      s1 = beta * std::exp(i * phase);
      s2 = std::conj(s1);

      const auto u = - 2 * std::real(s1);
      const auto v = std::real(s1 * s2);

      polynomial = Z.pow<double>(2) + u * Z + v;

#ifdef SHOW_DEBUG_LOG
      LOG_DEBUG << "sigma[X] = " << sigma << endl;
      LOG_DEBUG << "s1 = " << this->s1() << endl;
      LOG_DEBUG << "s2 = " << this->s2() << endl;
      LOG_DEBUG << "u = " << this->u() << endl;
      LOG_DEBUG << "v = " << this->v() << endl;
#endif
    }

    auto s1() const -> const double& { return roots[0]; }
    auto s2() const -> const double& { return roots[1]; }
    auto u() const -> const double& { return sigma[1]; }
    auto v() const -> const double& { return sigma[0]; }

    UnivariatePolynomial<double> polynomial;
    std::array<std::complex<double>, 2> roots;
  }


  struct AuxiliaryVariables
  {
    //! @{
    //! @brief Polynomial euclidean divisions.
    std::pair<UnivariatePolynomial<double>, UnivariatePolynomial<double>>
        P_div_sigma;
    std::pair<UnivariatePolynomial<double>, UnivariatePolynomial<double>>
        K0_div_sigma;
    //! @}

    double P_s1, P_s2;
    double a, b;

    double K0_s1, K0_s2;
    double c, d;

    auto update_target_polynomial_aux_vars(const QuadraticFactor& sigma) -> void
    {
      P_div_sigma = P / sigma.polynomial;
      const auto& [Q_P, R_P] = P_div_sigma;

      const auto& [s1, s2] = sigma.roots;
      P_s1 = R_P(s1);
      P_s2 = R_P(s2);

      b = P_remainder[1];
      a = P_remainder[0] - b * sigma.u();

#ifdef SHOW_DEBUG_LOG
      LOG_DEBUG << "P_remainder = " << P_remainder << endl;
      LOG_DEBUG << "P(s1) = "
                << "P(" << s1 << ") = " << P_s1 << endl;
      LOG_DEBUG << "P(s2) = "
                << "P(" << s2 << ") = " << P_s2 << endl;
      LOG_DEBUG << "a = " << a << endl;
      LOG_DEBUG << "b = " << b << endl;
#endif
    }

    auto update_shift_polynomial_aux_vars(const QuadraticFactor& sigma) -> void
    {
      K0_div_sigma = K0 / sigma.polynomial;
      const auto& [Q_K0, R_K0] = K0_div_sigma;

      K_s1 = R_K0(s1);
      K_s2 = R_K0(s2);

      d = R_K0[1];
      c = R_K0[0] - d * sigma.u();

#ifdef SHOW_DEBUG_LOG
      LOG_DEBUG << "R_K0 = " << R_K0 << endl;
      LOG_DEBUG << "K0(s1) = "
                << "K0(" << s1 << ") = " << K0_s1 << endl;
      LOG_DEBUG << "K0(s2) = "
                << "K0(" << s2 << ") = " << K0_s2 << endl;
      LOG_DEBUG << "c = " << c << endl;
      LOG_DEBUG << "d = " << d << endl;
#endif
    }
  };


  auto initial_shift_polynomial(const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>
  {
    return derivative(P) / P.degree();
  }

  auto next_zero_shift_polynomial(const UnivariatePolynomial<double>& K0,
                                  const UnivariatePolynomial<double>& P)
  {
    // The two formula below are identical but the former might not very stable
    // numerically...
    //
    // auto K1 = ((K0 - (K0(0) / P(0)) * P) / Z).first;
    auto K1 = ((K0 - K0(0)) / Z).first - (K0(0) / P(0)) * (P / Z).first;

    return K1;
  }


  auto next_linear_shift_polynomial(const UnivariatePolynomial<double>& K0,
                                    const UnivariatePolynomial<double>& P,
                                    double s_i, double P_si, double K0_si)
      -> UnivariatePolynomial<double>
  {
    auto K1 = ((K0 - K0_si) / (Z - s_i)).first -
              (K0_si / P_si) * ((P - P_si) / (Z - s_i)).first;

    /*
     * p(s) * k0(z) - p(s) * k0(s) - k0(s) * p(z) + k0(s) * p(s)
     * p(s) * k0(z)                - k0(s) * p(z)
     * |p (s)  p (z)|
     * |k0(s)  k0(z)|
     * */

    return K1;
  }


  auto next_quadratic_shift_polymomial(const UnivariatePolynomial<double>& P,
                                       const QuadraticFactor& sigma,
                                       const AuxiliaryVariables& aux)
    -> UnivariatePolynomial<double>
  {
    const auto& a = aux.a;
    const auto& b = aux.b;
    const auto& c = aux.c;
    const auto& d = aux.d;

    const auto& u = sigma.u();
    const auto& v = sigma.v();

    const auto c0 = b * c - a * d;
    const auto c1 = (a * a + u * a * b + v * b * b) / c0;
    const auto c2 = (a * c + u * a * d + v * b * d) / c0;

    auto K1 = c1 * Q_K0 + (Z - c2) * Q_P + b;

#ifdef SHOW_DEBUG_LOG
    LOG_DEBUG << "K1 = " << K1 << std::endl;
#endif

    return K1;
  }


  auto next_quadratic_factor(QuadraticFactor& sigma,
                             const UnivariatePolynomial<double>& P,
                             const UnivariatePolynomial<double>& K0,
                             const AuxiliaryVariables& aux)
    -> QuadraticFactor
  {
    const auto& a = aux.a;
    const auto& b = aux.b;
    const auto& c = aux.c;
    const auto& d = aux.d;

    const auto& u = sigma.u();
    const auto& v = sigma.v();

    const auto b1 = -K0[0] / P[0];
    const auto b2 = -(K0[1] + b1 * P[1]) / P[0];

    const auto a1 = b * c - a * d;
    const auto a2 = a * c + u * a * d + v * b * d;

    const auto c2 = b1 * a2;
    const auto c3 = b1 * b1 * (a * a + u * a * b + v * b * b);

    const auto c4 = v * b2 * a1 - c2 - c3;
    const auto c1 = c * c + u * c * d + v * d * d +
                    b1 * (a * c + u * b * c + v * b * d) - c4;

    const auto delta_u = -(u * (c2 + c3) + v * (b1 * a1 + b2 * a2)) / c1;
    const auto delta_v = v * c4 / c1;

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

    LOG_DEBUG << "delta_u = " << delta_u << endl;
    LOG_DEBUG << "delta_v = " << delta_v << endl;
#endif

    auto sigma_next = sigma;
    sigma_next.polynomial[0] += delta_v;
    sigma_next.polynomial[1] += delta_u;

    return sigma_next;
  }


} /* namespace DO::Sara */
