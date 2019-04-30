#include <DO/Sara/Core/Math/JenkinsTraub.hpp>

#include <complex>
#include <ctime>
#include <memory>


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


  namespace detail {

    TargetPolynomial::TargetPolynomial(const UnivariatePolynomial<double>& P)
      : P{P}
    {
    }

    auto TargetPolynomial::evaluate_at_divisor_roots(
        const QuadraticRealDivisor& sigma) -> void
    {
      P_s1 = P(sigma.s1);
      P_s2 = P(sigma.s2);
    }


    QuadraticRealDivisor::QuadraticRealDivisor(
        const UnivariatePolynomial<double>& P)
      : P{P}
    {
      beta = compute_moduli_lower_bound(P);
      initialize_randomly();
    }

    QuadraticRealDivisor::operator const UnivariatePolynomial<double>&() const
    {
      return sigma;
    }

    auto QuadraticRealDivisor::u() const -> const double&
    {
      return sigma[1];
    }

    auto QuadraticRealDivisor::v() const -> const double&
    {
      return sigma[0];
    }

    auto QuadraticRealDivisor::initialize_randomly() -> void
    {
      constexpr auto i = std::complex<double>{0, 1};

      const auto phase = dist(rd);
      s1 = beta * std::exp(i * phase);
      s2 = std::conj(s1);

      sigma = Z.pow<double>(2) - 2 * std::real(s1) * Z + std::real(s1 * s2);
    }

    auto QuadraticRealDivisor::update(const ShiftPolynomial& K) -> void
    {
      const auto& K0 = K.K0;
      const auto& a = K.a;
      const auto& b = K.b;
      const auto& c = K.c;
      const auto& d = K.d;
      const auto& u = this->u();
      const auto& v = this->v();

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

      // sigma = Z.pow<double>(2) + (u + delta_u) * Z + v + delta_v;
      sigma[0] = v + delta_v;
      sigma[1] = u + delta_u;
      sigma[2] = 1.;
    }


    ShiftPolynomial::ShiftPolynomial(const UnivariatePolynomial<double>& P)
      : P{P}
    {
    }

    auto ShiftPolynomial::evaluate_at_divisor_roots(const QuadraticRealDivisor& sigma)
        -> void
    {
      K0_s1 = K0(sigma.s1);
      K0_s2 = K0(sigma.s2);
    }

    auto ShiftPolynomial::calculate_coefficients_of_linear_remainders(
        const TargetPolynomial& P,          //
        const QuadraticRealDivisor& sigma)  //
        -> void
    {
      const auto& s1 = sigma.s1;
      const auto& s2 = sigma.s2;

      const auto& P_s1 = P.P_s1;
      const auto& P_s2 = P.P_s2;

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
    }

    auto ShiftPolynomial::update(const QuadraticRealDivisor& sigma) -> void
    {
      const auto& v = sigma.u();
      const auto& u = sigma.v();

      P_r = b * (Z + u) + a;
      K0_r = c * (Z + u) + d;

      Q_P = ((P - P_r) / sigma).first;
      Q_K0 = ((K0 - K0_r) / sigma).first;

      const auto c0 = b * c - a * d;
      const auto c1 = (a * a + u * a * b + v * b * b) / c0;
      const auto c2 = (a * c + u * a * d + v * b * d) / c0;

      K1 = c1 * Q_K0 + (Z - c2) * Q_P + b;
    }

  }  /* namespace detail */


  //! Used for stage 2.
  struct WeakConvergenceTest
  {
    enum ConvergenceType : std::uint8_t
    {
      NoConvergence = 0,
      LinearFactor = 1,
      QuadraticFactor = 2
    };

    template <typename T>
    inline auto impl(const std::array<T, 3>& seq) const -> bool
    {
      return std::abs(seq[1] - seq[0]) <= std::abs(seq[0]) / 2 &&
             std::abs(seq[2] - seq[1]) <= std::abs(seq[1]) / 2;
    }

    auto linear_convergence(const std::array<double, 3>& t) const -> bool
    {
      return impl<double>(t);
    }

    auto
    quadratic_convergence(const std::array<std::complex<double>, 3>& t) const
        -> bool
    {
      return impl<std::complex<double>>(t);
    }
  };


  //! Accentuate smaller zeros.
  auto stage1(UnivariatePolynomial<double>& K0,       //
              const UnivariatePolynomial<double>& P,  //
              int M)                                  //
      -> void
  {
    K0 = K0_polynomial(P);
    for (int i = 1; i < M; ++i)
      K0 = K1_no_shift_polynomial(K0, P);
  }

} /* namespace Sara */
} /* namespace DO */
