#pragma once

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>


namespace DO::Sara {

//! @brief Random Sample Consensus algorithm from Fischler and Bolles 1981.
template <typename Estimator, typename Distance, typename T>
auto ransac(const TensorView_<int, 2>& matches,      //
            const TensorView_<T, 2>& p1,             //
            const TensorView_<T, 2>& p2,             //
            Estimator estimator, Distance distance,  //
            int num_samples, int tuple_size, T dist_threshold)
{
  // ==========================================================================
  // Point coordinates.
  const auto P1 = homogeneous(p1);
  const auto P2 = homogeneous(p2);

  // Normalization transformation.
  const auto T1 = compute_normalizer(P1);
  const auto T2 = compute_normalizer(P2);

  // Normalized image coordinates.
  const auto P1n = apply_transform(T1, P1);
  const auto P2n = apply_transform(T2, P2);

  // ==========================================================================
  // Generate random samples for RANSAC.
  constexpr auto N = num_samples;
  constexpr auto L = tuple_size;

  // M = list of matches.
  const auto& M = matches;
  const auto card_M = M.size(0);

  // S is the list of N groups of L matches.
  const auto S = random_samples(N, L, card_M);
  // Remap each match index 'm' to a pair of point indices '(x, y)'.
  const auto I = to_point_indices(S, M);

  // Remap each pair of point indices to a pair of point coordinates.
  const auto p = to_coordinates(I, P1, P2).transpose({0, 2, 1, 3});

  // Normalized coordinates.
  const auto Pn = to_coordinates(I, P1n, P2n).transpose({0, 2, 1, 3});

  // Helper function.
  const auto count_inliers = [&](const auto& model) {
    auto num_inliers = 0;
    for (auto m = 0; m < M.size(0); ++m)
    {
      const auto i = M(m, 0);
      const auto j = M(m, 1);

      const auto xi = P1[i][0].colmajor_view().matrix();
      const auto yi = P2[i][1].colmajor_view().matrix();

      if (distance(model, xi, yi) < dist_threshold)
        ++num_inliers;
    }
    return num_inliers;
  };

  auto model_best = typename Estimator::model_type{};
  auto num_inliers_best = 0;

  for (auto n = 0; n < N; ++n)
  {
    // Normalized point coordinates.
    const Matrix<double, 3, L> Xn = Pn[n][0].colmajor_view().matrix();
    const Matrix<double, 3, L> Yn = Pn[n][1].colmajor_view().matrix();

    // Estimate the normalized models.
    auto models = estimator(Xn, Yn);

    // Unnormalize the essential matrices.
    for (auto& model : models)
    {
      model = model.normalized();
      model = (T2.transpose() * model * T1).normalized();
    }

    for (const auto& model: models)
    {
      const auto num_inliers = count_inliers(model);
      if (num_inliers > num_inliers_best)
      {
        model_best = model;
        num_inliers_best = num_inliers;

        SARA_CHECK(model_best);
        SARA_CHECK(num_inliers_best);
      }
    }
  }

  return std::make_tuple(model_best, num_inliers_best);
}

} /* namespace DO::Sara */
