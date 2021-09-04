// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Numpy.hpp>
#include <DO/Sara/Core/Random.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/Normalizer.hpp>


namespace DO::Sara {

  //! @ingroup MultiViewGeometry
  //! @defgroup RANSAC RANSAC
  //! @{

  template <typename MinimalSolver, typename InlierPredicate>
  struct RANSAC
  {
    static constexpr auto DataPointDimension =
        MinimalSolver::DataPointDimension;
    static constexpr auto MinimalSubsetCardinality =
        MinimalSolver::MinimalSubsetCardinality;

    using Real = typename MinimalSolver::real_type;
    using MinimalDataPoints = typename MinimalSolver::MinimalDataPoints;
    using DataPoints = TensorView_<Real, 2>;
    using Model = typename MinimalSolver::Model;

    auto operator()(const DataPoints& p1,  //
                    const DataPoints& p2,  //
                    Real threshold,        //
                    int num_samples)       //
        -> std::tuple<Model, Tensor_<bool, 1>, Tensor_<int, 1>>
    {
      // Normalization transformation.
      auto normalizer = Normalizer<Model>{p1, p2};

      // Normalized image coordinates.
      const auto [p1n, p2n] = normalizer.normalize(p1, p2);

      // Generate random samples for RANSAC.
      const auto& N = num_samples;

      // M = list of matches.
      const auto card_M = p1.cols();

      if (card_M < Estimator::num_points)
        throw std::runtime_error{"Not enough matches!"};

      // S is the list of N groups of L matches drawn randomly.
      const auto S = random_samples(N, L, card_M);

      // Remap every match index 'm' to a pair of point indices '(x, y)' that
      // are in the list of random samples.
      const auto I = to_point_indices(S, M);

      // Remap every pair of (sampled) point indices to a pair of point
      // coordinates.
      const auto p = to_coordinates(I, p1, p2).transpose({0, 2, 1, 3});

      // Normalize the coordinates coordinates.
      const auto pn = to_coordinates(I, p1n, p2n).transpose({0, 2, 1, 3});

      // For the inliers count.
      auto coords_matched = Tensor_<double, 3>{{2, card_M, 3}};
      auto p1_matched_mat = coords_matched[0].colmajor_view().matrix();
      auto p2_matched_mat = coords_matched[1].colmajor_view().matrix();
      {
        const auto mindices = range(card_M);
        const auto p1_mat = p1.colmajor_view().matrix();
        const auto p2_mat = p2.colmajor_view().matrix();
        std::for_each(std::begin(mindices), std::end(mindices), [&](int m) {
          p1_matched_mat.col(m) = p1_mat.col(M(m, 0));
          p2_matched_mat.col(m) = p2_mat.col(M(m, 1));
        });
      }

      auto model_best = Model{};

      auto num_inliers_best = 0;
      auto subset_best = Tensor_<int, 1>{Estimator::num_points};
      auto inliers_best = Tensor_<bool, 1>{card_M};

      for (auto n = 0; n < N; ++n)
      {
        // Normalized point coordinates.
        const MinimalDataPoints xn = pn[n][0].colmajor_view().matrix();
        const MinimalDataPoints yn = pn[n][1].colmajor_view().matrix();

        // Estimate the normalized models.
        auto models = estimator(xn, yn);

        // Unnormalize the models.
        for (auto& model : models)
          model.matrix() = normalizer.denormalize(model);

        for (const auto& model : models)
        {
          // Count the inliers.
          inlier_predicate.set_model(model);
          const auto inliers = inlier_predicate(p1_matched_mat, p2_matched_mat);
          const auto num_inliers = static_cast<int>(inliers.count());

          if (num_inliers > num_inliers_best)
          {
            num_inliers_best = num_inliers;
            model_best = model;
            inliers_best.flat_array() = inliers;
            subset_best = S[n];

            SARA_CHECK(model_best);
            SARA_CHECK(num_inliers);
            SARA_CHECK(subset_best.row_vector());
          }
        }
      }

      return std::make_tuple(model_best, inliers_best, subset_best);
    }
  };

} /* namespace DO::Sara */
