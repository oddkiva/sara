// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Random.hpp>
#include <DO/Sara/RANSAC/Utility.hpp>

#include <concepts>
#include <vector>


namespace DO::Sara {

  template <typename T>
  concept DataPointListConcept = requires(T data_points)
  {
    typename T::value_type;

    // clang-format off
    { data_points[std::declval<int>()] }
      -> std::convertible_to<typename T::value_type>>;
    // clang-format on
  };

  template <typename T>
  concept MinimalSolverConcept = requires(T solver)
  {
    T::num_points;
    T::num_models;
    typename T::data_point_type;
    typename T::model_type;

    // clang-format off
    { solver(std::declval<T::data_point_type>()) }
      -> std::same_as<std::vector<typename T::model_type>>;
    // clang-format on
  };


  //! @brief Random Sample Consensus algorithm from Fischler and Bolles 1981.
  //! batched computations and more generic API.
  template <DataPointListConcept DataPointList,  //
            MinimalSolverConcept ModelSolver,    //
            typename InlierPredicateType,
            typename DataNormalizer>
  auto ransac(
      const DataPointList& data_points,                                     //
      ModelSolver solver,                                                   //
      InlierPredicateType inlier_predicate,                                 //
      const std::size_t num_samples,                                        //
      const std::optional<DataNormalizer>& data_normalizer = std::nullopt)  //
      -> std::tuple<typename ModelSolver::model_type,                       //
                    Tensor_<bool, 1>,                                       //
                    Tensor_<int, 1>>                                        //
  {
    // X = data points.
    const auto& X = data_points;

    // Xn = normalized data points.
    const auto& Xn = data_normalizer.has_value()  //
                         ? data_normalizer->normalize(X)
                         : X;

    // Define cardinality variables.
    const auto N = static_cast<int>(num_samples);
    static constexpr auto L = ModelSolver::num_points;
    const auto& card_X = X.size(0);
    if (card_X < ModelSolver::num_points)
      throw std::runtime_error{"Not enough data points!"};

    const auto minimal_index_subsets = random_samples(N, L, card_X);
    const auto Xn_sampled = from_index_to_point(minimal_index_subsets, Xn);

    // For the inliers count.
    auto model_best = typename ModelSolver::model_type{};

    auto num_inliers_best = 0;
    auto subset_best = Tensor_<int, 1>{L};
    auto inliers_best = Tensor_<bool, 1>{card_X};

    for (auto n = 0; n < N; ++n)
    {
      // Estimate the candidate models with the normalized data.
      auto candidate_models = solver(Xn_sampled[n]);

      // Denormalize the candiate models from the data.
      if (data_normalizer.has_value())
        std::for_each(candidate_models.begin(), candidate_models.end(),
                      [&data_normalizer](auto& model) {
                        data_normalizer->denormalize(model);
                      });

      // Count the inliers.
      for (const auto& model : candidate_models)
      {
        // Count the inliers.
        inlier_predicate.set_model(model);
        const auto inliers = inlier_predicate(X);
        const auto num_inliers = static_cast<int>(inliers.count());

        if (num_inliers > num_inliers_best)
        {
          num_inliers_best = num_inliers;
          model_best = model;
          inliers_best.flat_array() = inliers;
          subset_best = minimal_index_subsets[n];

          SARA_CHECK(model_best);
          SARA_CHECK(num_inliers);
          SARA_CHECK(subset_best.row_vector());
        }
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
