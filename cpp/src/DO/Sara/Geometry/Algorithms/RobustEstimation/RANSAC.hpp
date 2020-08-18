// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Random.hpp>


namespace DO::Sara {

  //! @ingroup Geometry
  //! @defgroup RANSAC RANSAC
  //! @{
  template <typename T>
  auto to_coordinates(const TensorView_<int, 2>& samples,
                      const TensorView_<T, 2>& points)
  {
    const auto num_samples = samples.size(0);
    const auto sample_size = samples.size(1);
    const auto point_dimension = points.size(1);
    const auto point_matrix = points.matrix();

    auto p = Tensor_<T, 3>{{num_samples, sample_size, point_dimension}};

    for (auto s = 0; s < num_samples; ++s)
    {
      auto subset_matrix = p[s].matrix();
      for (auto k = 0; k < sample_size; ++k)
        subset_matrix.row(k) = point_matrix.row(samples(s, k));
    }

    return p;
  }

  //! @brief Random Sample Consensus algorithm from Fischler and Bolles 1981.
  //! batched computations and more generic API.
  template <typename T, typename ModelSolver, typename InlierPredicateType>
  auto ransac(const TensorView_<T, 2>& points,         //
              ModelSolver solver,                      //
              InlierPredicateType inlier_predicate,    //
              std::size_t num_samples)                 //
      -> std::tuple<typename ModelSolver::model_type,  //
                    Tensor_<bool, 1>,                  //
                    Tensor_<int, 1>>                   //
  {
    // Generate random samples for RANSAC.
    const auto& N = num_samples;
    constexpr auto L = ModelSolver::num_points;

    // P = list of points.
    const auto& P = points;
    const auto card_P = P.size(0);
    if (card_P < ModelSolver::num_points)
      throw std::runtime_error{"Not enough data points!"};

    // S is the list of N random elemental subsets, each of them having
    // cardinality L.
    const auto S = random_samples(N, L, card_P);

    // Remap every (sampled) point indices to point coordinates.
    const auto p = to_coordinates(S, points);

    // For the inliers count.
    auto model_best = typename ModelSolver::model_type{};

    auto num_inliers_best = 0;
    auto subset_best = Tensor_<int, 1>{L};
    auto inliers_best = Tensor_<bool, 1>{card_P};

    for (auto n = 0u; n < N; ++n)
    {
      // Estimate the normalized models.
      auto model = solver(p[n].matrix());

      // Count the inliers.
      inlier_predicate.set_model(model);
      const auto inliers = inlier_predicate(points.matrix());
      const auto num_inliers = static_cast<int>(inliers.count());

      if (num_inliers > num_inliers_best)
      {
        num_inliers_best = num_inliers;
        model_best = model;
        inliers_best.flat_array() = inliers;
        subset_best = S[n];

#ifdef DEBUG
        SARA_CHECK(model_best);
        SARA_CHECK(num_inliers);
        SARA_CHECK(subset_best.row_vector());
#endif
      }
    }

    return std::make_tuple(model_best, inliers_best, subset_best);
  }


  //! @brief Set the distance relative to the model parameters.
  template <typename Distance>
  struct InlierPredicate
  {
    using distance_type = Distance;
    using scalar_type = typename Distance::scalar_type;

    distance_type distance;
    scalar_type error_threshold;

    inline void set_model(const typename Distance::model_type& model)
    {
      distance = Distance{model};
    }

    //! @brief Calculate inlier predicate on a batch of correspondences.
    template <typename Mat>
    inline auto operator()(const Mat& x) const -> Eigen::Array<bool, 1, Dynamic>
    {
      return distance(x).array() < error_threshold;
    }
  };

  //! @}

}  // namespace DO::Sara
