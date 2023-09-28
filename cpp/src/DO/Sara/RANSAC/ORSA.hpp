#pragma once

#include <DO/Sara/RANSAC/AContrario.hpp>


namespace DO::Sara {

  // Optimal Random SAmpling (ORSA) acts like vanilla RANSAC at the beginning
  // of the sampling.
  // - Start sampling normally like RANSAC
  // - If we identify a model whose NFA is lower than a threshold ε, we have
  //   detected a near-optimal subset of inlier S.
  //   Quoting Rabin's thesis, then only sample minimal subsets in this subset
  //   S.
  template <typename DataPoint, typename MinimalSolver,
            typename NormalizedDistance>
  struct ORSA
  {
    auto estimate_noise_scale(
        const std::vector<IndexedResidual>& r,
        const std::vector<double>& log_nfas,
        const std::vector<double>::const_iterator log_nfa_min_it) const
    {
      // Since the residuals are sorted, we can identify the last inlier and
      // the first outlier.
      const auto last_inlier = r.begin() +                      //
                               std::distance(log_nfas.begin(),  //
                                             log_nfa_min_it);
      const auto first_outlier = last_inlier + 1;
      // We estimate the inlier/outlier threshold as the mid-point between
      // the last inlier residual and the first outlier residual.
      return first_outlier == r.end()
                 ? last_inlier->value
                 : 0.5 * (last_inlier->value + first_outlier->value);
    }

    // Step 1.
    auto find_inliers_and_noise(const TensorView_<DataPoint, 1>& X,  //
                                MinimalSolver&& solver,              //
                                const NormalizedDistance& d,         //
                                const double ε,  // the log-NFA threshold
                                const int max_num_iterations) const  //
    {
      const auto card_X = static_cast<int>(X.size());
      if (card_X < MinimalSolver::num_points)
        throw std::runtime_error{"Not enough data points!"};

      // S is the list of N random elemental subsets, each of them having
      // cardinality L.
      // Generate random samples for RANSAC.
      static constexpr auto L = MinimalSolver::num_points;
      const auto& N = max_num_iterations;
      const auto S = random_samples(N, L, card_X);

      // Remap every (sampled) point indices to point coordinates.
      const auto p = to_coordinates(S, X);

      // For the inliers count.
      auto θ_best = typename MinimalSolver::model_type{};
      auto log_nfa_best = std::numeric_limits<double>::infinity();
      auto residuals_best = std::vector<IndexedResidual>{};
      // The noise scale, i.e., the inlier/outlier threshold is also estimated.
      auto σ = std::numeric_limits<double>::quiet_NaN();

      auto found_optimal_inlier_subset = false;
      for (auto n = 0; n < N; ++n)
      {
        // Estimate the model from the minimal subset p.
        const auto θ = solver(p[n].matrix());

        // Calculate the sorted list of residuals.
        auto r = normalized_residuals(S, θ, X, d);

        // Transform the residuals as NFAs.
        // Notice the reference to perform in-place transformations.
        auto log_nfas = std::vector<double>(r.size());
        for (auto i = L; i < card_X; ++i)
        {
          const auto& candidate_subset_size = i;
          log_nfas[i] = log_nfa<MinimalSolver>(card_X,                 //
                                               candidate_subset_size,  //
                                               r[i].value);
        }

        const auto log_nfa_min_it = std::min_element(log_nfas.begin(),  //
                                                     log_nfas.end());

        if (*log_nfa_min_it < log_nfa_best)
          log_nfa_best = *log_nfa_min_it;

        if (*log_nfa_min_it >= ε)
          continue;

        // OK, it's good but are we dealing with a degenerate configuration?
        // In the epipolar geometry problem, we can check with DEGENSAC?
        if constexpr (MinimalSolver::postcondition_required)
        {
          const auto degeneracy_fixed = false;  // fix_degeneracy(model, S);
          if (degeneracy_fixed)
          {
            θ_best = θ;
            found_optimal_inlier_subset = true;
            residuals_best = r;
            σ = estimate_noise_scale(r, log_nfas, log_nfa_min_it);
            break;
          }
        }
        else
        {
          θ_best = θ;
          found_optimal_inlier_subset = true;
          residuals_best = r;
          σ = estimate_noise_scale(r, log_nfas, log_nfa_min_it);
          break;
        }
      }

      if (!found_optimal_inlier_subset)
        return std::nullopt;

      return std::make_optional(std::make_tuple(θ_best,          //
                                                residuals_best,  //
                                                log_nfa_best, σ));
    }

    // Step 2.
    auto polish_model(const std::vector<std::size_t>&) const
    {
      // Find the best model by subsampling only within the set of inliers.
      // TODO
      //
      // I think it would be better to decompose the fundamental matrix and
      // optimize it globally...
    }
  };

}  // namespace DO::Sara
