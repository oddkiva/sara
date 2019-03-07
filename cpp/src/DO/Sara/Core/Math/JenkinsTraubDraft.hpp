
  // Fixed-shift process.
  struct Stage2
  {
    //! @{
    //! @brief parameters.
    int M{5};
    int L{20};
    //! @}


    //! Determine convergence type.
    auto stage2(const UnivariatePolynomial<double>& P) -> ConvergenceType
    {
      const auto beta = compute_moduli_lower_bound(P);

      auto cvg_type = ConvergenceType{};

      // Stage 2 must be able to determine the convergence.
      while (cvg_type == NoConvergence)
      {
        // Choose roots randomly on the circle of radius beta.
        form_quadratic_divisor_sigma();

        // Do it only once.
        evaluate_polynomial_at_divisor_roots();

        auto t = std::array<std::complex<double>, 3>{{0., 0., 0.}};
        auto v = std::array<double, 3>{0, 0, 0};

        // Determine convergence type.
        for (int i = M; i < L; ++i)
        {
          evaluate_shift_polynomial_at_divisor_roots();

          calculate_coefficients_of_linear_remainders();
          calculate_next_shift_polynomial();

          t[0] = t[1];
          t[1] = t[2];
          t[2] = s1 - P_s1 / K0_s1;

          v[0] = v[1];
          v[1] = v[2];
          v[2] = sigma[2];

          K0 = K1;

          if (i < M + 3)
            continue;

          if (check_convergence_linear_factor(t))
          {
            cvg_type = LinearFactor;
            break;
          }

          if (check_convergence_quadratic_factor(v))
          {
            cvg_type = QuadraticFactor;
            break;
          }
        }

        // The while loop will keep going if cvg_type is NoConvergence.
      }
    }

    auto stage3() -> void
    {
      auto t = std::array<std::complex<double>, 3>{{0., 0., 0.}};
      auto v = std::array<double, 3>{0, 0, 0};

      evaluate_polynomial_at_divisor_roots();
      evaluate_shift_polynomial_at_divisor_roots();
      auto s_i = std::real(s1 - P_s1 / K0_s1);
      auto v_i = sigma[2];

      // Determine convergence type.
      while (true)
      {
        calculate_coefficients_of_linear_remainders();
        calculate_next_shift_polynomial();
        calculate_next_quadratic_divisor();

        if (cvg_type == LinearFactor)
        {
          s_i -= -P(s_i) / K1(s_i);
          // Check convergence.
        }

        if (cvg_type == QuadraticFactor)
        {
          v_i = sigma[2];
          // Check convergence.
        }

        // Update K0.
        K0 = K1;
        evaluate_polynomial_at_divisor_roots();
        evaluate_shift_polynomial_at_divisor_roots();
      }

      // TODO: deflate polynomial and restart again.
    }
  };
