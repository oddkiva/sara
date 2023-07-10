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

#include <DO/Sara/RANSAC/RANSAC.hpp>


namespace DO::Sara::v2 {

  //! @brief Random Sample Consensus algorithm from Fischler and Bolles 1981.
  //! batched computations and more generic API.
  template <DataPointListConcept DataPointList,  //
            MinimalSolverConcept ModelSolver,    //
            typename InlierPredicateType>
  auto ransac(const DataPointList& data_points,      //
              ModelSolver solver,                    //
              InlierPredicateType inlier_predicate,  //
              const int num_iterations_max,          //
              const double confidence = 0.999,       //
              const std::optional<Normalizer<typename ModelSolver::model_type>>&
                  data_normalizer = std::nullopt,
              bool verbose = false)                    //
      -> std::tuple<typename ModelSolver::model_type,  //
                    Tensor_<bool, 1>,                  //
                    Tensor_<int, 1>>                   //
  {
    // X = data points.
    const auto& X = data_points;

    // Xn = normalized data points.
    const auto& Xn = data_normalizer.has_value()  //
                         ? data_normalizer->normalize(X)
                         : X;

    // Define cardinality variables.
    const auto& card_X = static_cast<int>(X.size());
    if (card_X < ModelSolver::num_points)
      throw std::runtime_error{"Not enough data points!"};

    const auto minimal_index_subsets =
        random_samples(num_iterations_max, ModelSolver::num_points, card_X);
    const auto Xn_sampled = from_index_to_point(minimal_index_subsets, Xn);

    auto model_best = typename ModelSolver::model_type{};
    auto num_inliers_best = 0;
    auto subset_best = Tensor_<int, 1>{ModelSolver::num_points};
    auto inliers_best = Tensor_<bool, 1>{card_X};

    auto inlier_ratio_current = 1. / card_X;
    auto num_iterations = std::numeric_limits<int>::max();

    const auto update_num_iterations = [&num_iterations, &inlier_ratio_current,
                                        &confidence, &num_iterations_max,
                                        &verbose]() {
      num_iterations = static_cast<int>(
          std::min(ransac_num_samples(inlier_ratio_current,
                                      ModelSolver::num_models, confidence),
                   static_cast<std::uint64_t>(num_iterations_max)));
      if (verbose)
      {
        SARA_CHECK(inlier_ratio_current);
        SARA_CHECK(num_iterations);
      }
    };
    update_num_iterations();

    for (auto n = 0; n < num_iterations; ++n)
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

          //
          inlier_ratio_current =
              num_inliers / static_cast<double>(data_points.size());
          update_num_iterations();

          if (verbose)
          {
            SARA_DEBUG << "n = " << n << "\n";
            SARA_DEBUG << "model_best = \n" << model_best << "\n";
            SARA_DEBUG << "num inliers = " << num_inliers << "\n";
            SARA_CHECK(subset_best.row_vector());
          }
        }
      }
    }

    return std::make_tuple(model_best, inliers_best, subset_best);
  }

}  // namespace DO::Sara::v2
