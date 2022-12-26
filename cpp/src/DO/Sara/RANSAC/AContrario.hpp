#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  template <typename T>
  constexpr auto combinatorial(const T& n, const int& k) -> T
  {
    static_assert(std::is_integral_v<T>);
    auto num = T{1};
    auto denum = T{1};
    for (auto i = 0; i < k; ++i)
      num *= n - i;
    for (auto i = 1; i <= k; ++i)
      denum *= i;

    return num / denum;
  }


  using DataPointIndices = std::vector<std::size_t>;

  struct IndexedResidual
  {
    std::size_t index;
    double residual;

    auto operator<(const IndexedResidual& other) const
    {
      return residual < other.residual;
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
                   [&d, &θ, &X](const auto& i) { return d(θ, X[i]); });

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
    static_assert(false, "Please implement the log NFA and not the NFA");
    return MinimalSolver::num_candidate_models *
           combinatorial(num_data_points, subset_size) *
           combinatorial(subset_size, MinimalSolver::num_points) *
           std::pow(normalized_residual, subset_size);
  }

  template <typename MinimalSolver, typename Model, typename DataPoint,
            typename NormalizedDistance>
  auto nfa(const DataPointIndices& S,        //
           const Model& θ,                   //
           const std::vector<DataPoint>& X,  //
           const NormalizedDistance& d) -> double
  {
    if (S.empty())
      return std::numeric_limits<double>::quiet_NaN();

    const auto α = *std::max_element(normalized_residuals(S, θ, X, d));
    return nfa(X.size(), S.size(), α);
  }

  // Optimal Random SAmpling (ORSA) acts like vanilla RANSAC at the beginning of
  // the sampling.
  // - If we identify a model whose NFA is lower than a threshold ε
  template <typename DataPoint, typename MinimalSolver,
            typename NormalizedDistance>
  auto orsa(const TensorView_<DataPoint, 1>& X,  //
            MinimalSolver&& solver,              //
            const NormalizedDistance& d,         //
            const double ε,                      // the log-NFA threshold
            const int max_num_iterations)        //
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
    auto model_best = typename MinimalSolver::model_type{};
    auto log_nfa_best = std::numeric_limits<double>::infinity();
    auto log_nfa_wrt_best_model = std::vector<IndexedResidual>{};

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
                                             r[i]->residual);
      }

      const auto log_nfa_min_it =
          std::min_element(log_nfas.begin(), log_nfas.end());
      const auto num_inliers = log_nfa_min_it - log_nfas.begin();

      if (*log_nfa_min_it < log_nfa_best)
        log_nfa_best = *log_nfa_min_it;

      if (*log_nfa_min_it >= ε)
        continue;

      // OK, it's good but are we dealing with a degenerate configuration?
      // In the epipolar geometry problem, we can check with DEGENSAC?
      if constexpr (MinimalSolver::postcondition_required)
      {
        // The estimated noise.
        auto sigma = r
      }
      else
      {
        found_optimal_inlier_subset = true;
        log_nfa_wrt_best_model = log_nfas;
        break;
      }
    }

    // Find the best model by subsampling only within the set of inliers.
    if (found_optimal_inlier_subset)
    {
      // TODO
      //
      // I think it would be better to decompose the fundamental matrix and
      // optimize it globally...
    }
  }


}  // namespace DO::Sara
