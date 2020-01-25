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

#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>


namespace DO::Sara {

auto to_tensor(const std::vector<Match>& matches) -> Tensor_<int, 2>
{
  auto match_tensor = Tensor_<int, 2>{int(matches.size()), 2};
  for (auto i = 0u; i < matches.size(); ++i)
    match_tensor[i].flat_array() << matches[i].x_index(), matches[i].y_index();
  return match_tensor;
}

auto extract_centers(const std::vector<OERegion>& features) -> Tensor_<float, 2>
{
  auto centers = Tensor_<float, 2>{{int(features.size()), 2}};
  auto mat = centers.matrix();

  for (auto i = 0; i < centers.size(0); ++i)
    mat.row(i) = features[i].center().transpose();

  return centers;
}

auto to_point_indices(const TensorView_<int, 2>& samples,
                      const TensorView_<int, 2>& matches)  //
    -> Tensor_<int, 3>
{
  const auto num_samples = samples.size(0);
  const auto sample_size = samples.size(1);

  auto point_indices = Tensor_<int, 3>{{num_samples, sample_size, 2}};
  for (auto s = 0; s < num_samples; ++s)
    for (auto m = 0; m < sample_size; ++m)
      point_indices[s][m].flat_array() = matches[samples(s, m)].flat_array();

  return point_indices;
}

} /* namespace DO::Sara */
