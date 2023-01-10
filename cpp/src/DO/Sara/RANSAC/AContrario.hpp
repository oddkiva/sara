#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  inline constexpr auto log_combinatorial(const int n, const int k) -> double
  {
    auto log_num = double{};
    auto log_denum = double{};
    for (auto i = 0; i < k; ++i)
      log_num += std::log(n - i);
    for (auto i = 1; i <= k; ++i)
      log_denum += std::log(i);
    return log_num - log_denum;
  }


  using DataPointIndices = std::vector<std::size_t>;

  struct IndexedResidual
  {
    std::size_t index;
    double value;

    auto operator<(const IndexedResidual& other) const
    {
      return value < other.value;
    }
  };


  // A Contrario methods coin the term "rigidity" to define how well a model
  // fits a subset of data points.
  //
  // I don't like this term since this term ties it to multiview geometry
  // problems.
  //
  // Two observations:
  // - This rigidity value is the maximal residual error between the model and
  //   the data points in the subset.
  // - More importantly, this residual error is normalized to ensure that the
  //   number of false alarms (NFA) gets a proper meaning.
  template <typename Model, typename DataPoint, typename NormalizedDistance>
  auto normalized_residuals(const DataPointIndices& S, const Model& θ,
                            const std::vector<DataPoint>& X,
                            const NormalizedDistance& d)
      -> std::vector<IndexedResidual>
  {
    // Calculate the list of residuals which we denote by $r$.
    auto r = std::vector<IndexedResidual>{};
    r.resize(X.size());
    std::transform(S.begin(), S.end(), r.begin(),
                   [&d, &θ, &X](const auto& i) -> IndexedResidual {
                     return {i, d(θ, X[i])};
                   });

    // Sort the residuals to ease the calculus of the NFA.
    std::sort(r.begin(), r.end());

    // TODO: remove the first seven points for which they must be 0.

    return r;
  }


  // Calculate the Number of False Alarms (NFA).
  template <typename MinimalSolver>
  auto log_nfa(const std::size_t num_data_points, const std::size_t subset_size,
               const double normalized_residual) -> double
  {
    return std::log(MinimalSolver::num_candidate_models) +
           log_combinatorial(num_data_points, subset_size) +
           log_combinatorial(subset_size, MinimalSolver::num_points) +
           std::log(normalized_residual) * subset_size;
  }

  template <typename MinimalSolver, typename Model, typename DataPoint,
            typename NormalizedDistance>
  auto log_nfa(const DataPointIndices& S,        //
               const Model& θ,                   //
               const std::vector<DataPoint>& X,  //
               const NormalizedDistance& d) -> double
  {
    if (S.empty())
      return std::numeric_limits<double>::quiet_NaN();

    const auto α = *std::max_element(normalized_residuals(S, θ, X, d));
    return log_nfa(X.size(), S.size(), α);
  }

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
