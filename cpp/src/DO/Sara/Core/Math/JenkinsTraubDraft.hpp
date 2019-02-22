
  struct Stage2
  {
    Stage2(const UnivariatePolynomial<double>& P,
           const UnivariatePolynomial<double>& sigma,
           std::complex<double>& s1)
      : P{P}
      , sigma{sigma}
      , s1{s1}
      , s2{std::conj(s1)}
      , P_s1{P(s1)}
      , P_s2{P(s2)}
      , K0{K0_polynomial(P)}
    {
      K0_s1 = K0(s1);
      K0_s2 = K0(s2);

      K0_s1 = K0(s1);
      K0_s2 = K0(s2);

      // See stage 2 formula (9.7) (page 563).
      Matrix4cd M;
      Vector4cd y;

      M << 1, -s2, 0,   0,
           0,   0, 1, -s2,
           1, -s1, 0,   0,
           0,   0, 1, -s1;

      y << P_s1, P_s2, K0_s1, K0_s2;
      Vector4cd x = M.colPivHouseholderQr().solve(y);

      const auto a = std::real(x[0]);
      const auto b = std::real(x[1]);
      const auto c = std::real(x[2]);
      const auto d = std::real(x[3]);

      const auto u = - std::real(s1 + s2);

      P_r = b * (Z + u) + a;
      K0_r = d * (Z + u) + c;
    }

    void operator()()
    {
      const auto beta = compute_moduli_lower_bound(P);
      const auto s1 = beta;           //* std::exp(i*rand());
      const auto s2 = std::conj(s1);  //* std::exp(i*rand());
      const auto sigma = sigma(s1);

      auto K0 = K0;
      auto K1 = K1_no_shift_polynomial_stage2(P, K0, sigma, s1, std::conj(s1));
      auto K2 = K1_no_shift_polynomial_stage2(P, K1, sigma, s1, std::conj(s1));
      auto K3 = K1_no_shift_polynomial_stage2(P, K2, sigma, s1, std::conj(s1));
      auto K4 = K1_no_shift_polynomial_stage2(P, K3, sigma, s1, std::conj(s1));

      auto u = std::real(s1 + s2);
      auto t0 = s1 - (a0 - b0 * s2) / (c0 - d0 * s2);
      auto t1 = s1 - (a1 - b1 * s2) / (c1 - d1 * s2);
      auto t2 = s1 - (a2 - b2 * s2) / (c2 - d2 * s2);

      for (int i = 0; i < L; ++i)
      {
        K0 = K1;
        K1 = K2;
        K2 = K1_stage2(P, K1, sigma, s1, std::conj(s1));

        t0 = t1;
        t1 = t2;
        t2 = s1 - P(s1) / K2(s1);

        auto sigma_0 = sigma(K0, K1, K2, s1, s2);
        auto sigma_1 = sigma(K1, K2, K3, s1, s2);
        auto sigma_2 = sigma(K2, K3, K4, s1, s2);

        const auto v0 = sigma_0[0];
        const auto v1 = sigma_1[0];
        const auto v2 = sigma_2[0];

        // Convergence to rho_1?
        if (std::abs(t1 - t0) <= 0.5 * t0 && std::abs(t2 - t1) <= 0.5 * t1)
          return t2;

        // Convergence of sigma(z) to (z - rho_1) * (z - rho_2)?
        if (std::abs(v1 - v0) <= 0.5 * v0 && std::abs(v2 - v1) <= 0.5 * v1)
          return t2;
      }
    }

    const UnivariatePolynomial<double>& P;
    const UnivariatePolynomial<double>& sigma;
    std::complex<double>& s1;
    std::complex<double> s2;
    std::complex<double> P_s1, P_s2;
    std::complex<double> K0_s1, K0_s2;

    UnivariatePolynomial<double> P_r;
    UnivariatePolynomial<double> K0_r;

    UnivariatePolynomial<double> K0;
    UnivariatePolynomial<double> K1;

    int L{20};
  };


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

