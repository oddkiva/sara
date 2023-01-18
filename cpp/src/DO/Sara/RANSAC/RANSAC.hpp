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
#include <DO/Sara/MultiViewGeometry/Geometry/Normalizer.hpp>
#include <DO/Sara/RANSAC/Utility.hpp>

#include <concepts>
#include <optional>
#include <vector>


namespace DO::Sara {

  //! @defgroup RANSAC RANSAC

  template <typename T>
  concept DataPointListConcept = requires(T data_points)
  {
    typename T::value_type;

    {data_points[std::declval<int>()]};
    {data_points.size()};
  };

  template <typename T>
  concept MinimalSolverConcept = requires(T solver)
  {
    T::num_points;
    T::num_models;
    typename T::data_point_type;
    typename T::model_type;

    // clang-format off
    { solver(std::declval<const typename T::data_point_type&>()) };
    // clang-format on
  };


  //! @brief Random Sample Consensus algorithm from Fischler and Bolles 1981.
  //! batched computations and more generic API.
  template <DataPointListConcept DataPointList,  //
            MinimalSolverConcept ModelSolver,    //
            typename InlierPredicateType>
  auto ransac(const DataPointList& data_points,      //
              ModelSolver solver,                    //
              InlierPredicateType inlier_predicate,  //
              const int num_samples,                 //
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
        random_samples(num_samples, ModelSolver::num_points, card_X);
    const auto Xn_sampled = from_index_to_point(minimal_index_subsets, Xn);

    auto model_best = typename ModelSolver::model_type{};
    auto num_inliers_best = 0;
    auto subset_best = Tensor_<int, 1>{ModelSolver::num_points};
    auto inliers_best = Tensor_<bool, 1>{card_X};

    for (auto n = 0; n < num_samples; ++n)
    {
      // Estimate the candidate models with the normalized data.
      auto candidate_models = solver(Xn_sampled[n]);

      // Denormalize the candiate models from the data.
      if (data_normalizer.has_value())
        std::for_each(candidate_models.begin(), candidate_models.end(),
                      [&data_normalizer](auto& model) {
                        model = data_normalizer->denormalize(model);
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

  template <typename Distance>
  struct InlierPredicate
  {
    Distance distance;
    double err_threshold;

    //! @brief Set the distance relative to the model parameters.
    template <typename Model>
    inline void set_model(const Model& model)
    {
      distance.set_model(model);
    }

    //! @brief Calculate inlier predicate on a batch of correspondences.
    template <typename Derived>
    inline auto operator()(const Eigen::MatrixBase<Derived>& x) const
        -> Eigen::Array<bool, 1, Dynamic>
    {
      return distance(x).array() < err_threshold;
    }

    //! @brief Check the inlier predicate on a batch of data points.
    template <typename T, int D>
    inline auto operator()(const PointList<T, D>& X) const
        -> Array<bool, 1, Dynamic>
    {
      return distance(X._data.colmajor_view().matrix()).array() < err_threshold;
    }

    //! @brief Check the inlier predicate on a list of correspondences.
    template <typename T>
    inline auto operator()(const PointCorrespondenceList<T>& m) const
        -> Array<bool, 1, Dynamic>
    {
      return distance(m._p1.colmajor_view().matrix(),
                      m._p2.colmajor_view().matrix())
                 .array() < err_threshold;
    }
  };

  template <DataPointListConcept DataPointList,  //
            MinimalSolverConcept ModelSolver,    //
            typename Distance>
  auto ransac(const DataPointList& data_points,       //
              ModelSolver solver, Distance distance,  //
              const int num_samples, const double err_threshold)
  {
    auto inlier_predicate = InlierPredicate<Distance>{};
    inlier_predicate.distance = distance;
    inlier_predicate.err_threshold = err_threshold;

    return ransac(data_points, solver, inlier_predicate, num_samples);
  }

  //! @brief From vanilla RANSAC
  inline auto ransac_num_samples(double inlier_ratio,
                                 int minimal_sample_cardinality,
                                 double confidence = 0.99) -> std::size_t
  {
    return std::log(1 - confidence) /
           std::log(1 - std::pow(inlier_ratio, minimal_sample_cardinality));
  }

  //! @}
}  // namespace DO::Sara
