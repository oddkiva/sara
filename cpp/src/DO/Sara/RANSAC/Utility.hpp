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


namespace DO::Sara {

  template <typename T, int N>
  auto from_index_pairs_to_point_pairs(const TensorView_<int, 2>& index_pairs,
                                       const TensorView_<T, N>& first,
                                       const TensorView_<T, N>& second)
      -> Tensor_<T, N + 1>
  {
    if (index_pairs.size(1) != 2)
      throw std::runtime_error{
          "The tensor of index pairs must contain 2 columns!"};

    const auto num_samples = index_pairs.size(0);
    const auto coords_dim = first.size(1);

    auto point_pairs = Tensor_<T, N + 1>{{num_samples, 2, coords_dim}};

    for (auto s = 0; s < num_samples; ++s)
    {
      const auto& i1 = index_pairs(s, 0);
      const auto& i2 = index_pairs(s, 1);

      point_pairs[s][0].flat_array() = first[i1].flat_array();
      point_pairs[s][1].flat_array() = second[i2].flat_array();
    }

    return point_pairs;
  }

  template <typename T, int D>
  auto from_index_to_point(const TensorView_<int, 1>& point_indices,
                           const TensorView_<T, D>& points) -> Tensor_<T, D>
  {
    const auto num_samples = point_indices.size(0);

    auto sample_sizes = points.sizes();
    sample_sizes(0) = num_samples;

    auto points_sampled = Tensor_<T, D>{sample_sizes};

    for (auto s = 0; s < num_samples; ++s)
      points_sampled[s].flat_array() = points[point_indices(s)].flat_array();

    return points_sampled;
  }

  template <typename T, int D>
  auto from_index_to_point(const TensorView_<int, 2>& point_indices,
                           const TensorView_<T, D>& points) -> Tensor_<T, D + 1>
  {
    const auto num_subsets = point_indices.size(0);
    const auto num_samples_per_subset = point_indices.size(1);

    const auto point_matrix = points.matrix();

    auto sample_sizes = typename Tensor_<T, D + 1>::vector_type{};
    sample_sizes << num_subsets, num_samples_per_subset,
        points.sizes().template tail<D - 1>();
    auto points_sampled = Tensor_<T, D + 1>{sample_sizes};

    for (auto n = 0; n < num_subsets; ++n)
    {
      const auto indices = point_indices[n];
      auto subset = points_sampled[n].matrix();
      for (auto s = 0; s < num_samples_per_subset; ++s)
        subset.row(s) = point_matrix.row(indices(s));
    }

    return points_sampled;
  }

}  // namespace DO::Sara
