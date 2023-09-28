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

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Geometry/Algorithms/RobustEstimation/PointList.hpp>
#include <DO/Sara/MultiViewGeometry/PointCorrespondenceList.hpp>


namespace DO::Sara {

  template <typename T, int D>
  auto from_index_to_point(const TensorView_<int, 1>& point_indices,
                           const TensorView_<T, D>& points) -> Tensor_<T, D>
  {
    const auto num_samples = point_indices.size(0);

    auto sample_sizes = points.sizes();
    sample_sizes(0) = num_samples;

    auto points_sampled = Tensor_<T, D>{sample_sizes};

    for (auto s = 0; s < num_samples; ++s)
      points_sampled[s] = points[point_indices(s)];

    return points_sampled;
  }

  template <typename T, int D>
  auto from_index_to_point(const TensorView_<int, 2>& point_indices,
                           const TensorView_<T, D>& points) -> Tensor_<T, D + 1>
  {
    const auto num_subsets = point_indices.size(0);
    const auto num_samples_per_subset = point_indices.size(1);

    auto sample_sizes = typename Tensor_<T, D + 1>::vector_type{};
    sample_sizes << num_subsets, num_samples_per_subset,
        points.sizes().template tail<D - 1>();
    auto points_sampled = Tensor_<T, D + 1>{sample_sizes};

    for (auto n = 0; n < num_subsets; ++n)
    {
      const auto indices = point_indices[n];
      auto subset = points_sampled[n];
      for (auto s = 0; s < num_samples_per_subset; ++s)
        subset[s] = points[indices(s)];
    }

    return points_sampled;
  }

  template <typename T, int D>
  auto from_index_to_point(const TensorView_<int, 2>& point_indices,
                           const PointList<T, D>& points) -> Tensor_<T, D + 1>
  {
    return from_index_to_point(point_indices, points._data);
  };

  template <typename T>
  auto from_index_to_point(const TensorView_<int, 2>& point_indices,
                           const PointCorrespondenceList<T>& correspondences)
      -> PointCorrespondenceSubsetList<T>
  {
    auto res = PointCorrespondenceSubsetList<T>{};
    res._p1 = from_index_to_point(point_indices, correspondences._p1);
    res._p2 = from_index_to_point(point_indices, correspondences._p2);
    return res;
  };

}  // namespace DO::Sara
