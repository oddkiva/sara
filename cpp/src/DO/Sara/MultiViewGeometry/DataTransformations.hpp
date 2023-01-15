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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Match/Match.hpp>


namespace DO::Sara {

  /*!
   *  @ingroup MultiViewGeometry
   *  @defgroup MultiviewDataTransformations Data Transformations
   *  @{
   */

  DO_SARA_EXPORT
  auto to_tensor(const std::vector<Match>& matches) -> Tensor_<int, 2>;

  DO_SARA_EXPORT
  auto extract_centers(const std::vector<OERegion>& features)
      -> Tensor_<float, 2>;

  DO_SARA_EXPORT
  auto to_point_indices(const TensorView_<int, 2>& samples,
                        const TensorView_<int, 2>& matches)  //
      -> Tensor_<int, 3>;

  template <typename T>
  auto to_coordinates(const TensorView_<int, 3>& point_indices,
                      const TensorView_<T, 2>& p1,
                      const TensorView_<T, 2>& p2)  //
  {
    const auto num_samples = point_indices.size(0);
    const auto sample_size = point_indices.size(1);
    static constexpr auto num_points = 2;
    const auto coords_dim = p1.size(1);

    auto p = Tensor_<T, 4>{{num_samples, sample_size, num_points, coords_dim}};

    for (auto s = 0; s < num_samples; ++s)
      for (auto m = 0; m < sample_size; ++m)
      {
        const auto p1_idx = point_indices(s, m, 0);
        const auto p2_idx = point_indices(s, m, 1);

        p[s][m][0].flat_array() = p1[p1_idx].flat_array();
        p[s][m][1].flat_array() = p2[p2_idx].flat_array();
      }

    return p;
  }

  //! @}

} /* namespace DO::Sara */
