#pragma once

#include <DO/Sara/Core/Tensor.hpp>

#include <optional>


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
  // - More importantly, this residual error is normalized in [0, 1] to ensure
  //   that the number of false alarms (NFA) has a proper statistical meaning.
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

}  // namespace DO::Sara
