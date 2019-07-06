#pragma once

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>


namespace DO::Sara {

//! @brief Random Sample Consensus algorithm from Fischler and Bolles 1981.
template <typename Estimator, typename Distance, typename T>
auto ransac(const TensorView_<int, 2>& matches,      //
            const TensorView_<T, 2>& p1,             //
            const TensorView_<T, 2>& p2,             //
            Estimator estimator, Distance distance,  //
            int num_samples, T dist_threshold)
{
  // Normalization transformation.
  const auto T1 = compute_normalizer(p1);
  const auto T2 = compute_normalizer(p2);

  // Normalized image coordinates.
  const auto p1n = apply_transform(T1, p1);
  const auto p2n = apply_transform(T2, p2);

  SARA_DEBUG << "p1n =\n" << p1n.matrix().topRows(10) << std::endl;
  SARA_DEBUG << "p2n =\n" << p2n.matrix().topRows(10) << std::endl;

  // ==========================================================================
  // Generate random samples for RANSAC.
  const auto& N = num_samples;
  constexpr auto L = Estimator::num_points;
  SARA_DEBUG << "L = " << L << std::endl;

  // M = list of matches.
  const auto& M = matches;
  const auto card_M = M.size(0);

  SARA_DEBUG << "M =\n" << M.matrix().topRows(10) << std::endl;
  SARA_DEBUG << "card_M = " << card_M << std::endl;

  if (card_M < Estimator::num_points)
    throw std::runtime_error{"Not enough matches!"};

  // S is the list of N groups of L matches.
  const auto S = random_samples(N, L, card_M);
  SARA_DEBUG << "S =\n" << S.matrix() << std::endl;

  // Remap each match index 'm' to a pair of point indices '(x, y)'.
  const auto I = to_point_indices(S, M);

  // Remap each pair of point indices to a pair of point coordinates.
  const auto p = to_coordinates(I, p1, p2).transpose({0, 2, 1, 3});

  // Normalized coordinates.
  const auto pn = to_coordinates(I, p1n, p2n).transpose({0, 2, 1, 3});


  // Helper function.
  const auto count_inliers = [&](const auto& model) {

    const auto p1mat = p1.colmajor_view().matrix();
    const auto p2mat = p2.colmajor_view().matrix();

    auto num_inliers = 0;
    for (auto m = 0; m < M.size(0); ++m)
    {
      const auto i = M(m, 0);
      const auto j = M(m, 1);

      const auto xi = p1mat.col(i);
      const auto yj = p2mat.col(j);

      if (distance(model, xi, yj) < dist_threshold)
        ++num_inliers;
    }
    return num_inliers;
  };

  auto model_best = typename Estimator::model_type{};
  auto num_inliers_best = 0;

  for (auto n = 0; n < N; ++n)
  {
    // Normalized point coordinates.
    const Matrix<double, 3, L> xn = pn[n][0].colmajor_view().matrix();
    const Matrix<double, 3, L> yn = pn[n][1].colmajor_view().matrix();

    // Estimate the normalized models.
    auto models = estimator(xn, yn);

    // Unnormalize the models.
    for (auto& model : models)
    {
      model.matrix() = model.matrix().normalized();
      model.matrix() = (T2.transpose() * model.matrix() * T1).normalized();
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
