#pragma once

#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures2.hpp>


namespace DO::Sara {

  template <typename T = double>
  struct SevenPointAlgorithm
  {
    static constexpr auto num_models = 3;
    static constexpr auto num_candidate_models = 3;

    auto operator()() const -> std::array<std::optional<Eigen::Matrix<T, 3, 3>>,
                                          num_candidate_models>;
  };


  template <typename EpipolarDistance>
  inline auto
  normalized_epipolar_residual(const std::vector<std::size_t>& subset,
                               const Eigen::Matrix3d& F,
                               const std::vector<Eigen::Vector4d>& matches,
                               const EpipolarDistance& distance)
      -> std::vector<double>
  {
    return normalized_residual(subset, F, matches, distance);
  }

}  // namespace DO::Sara
