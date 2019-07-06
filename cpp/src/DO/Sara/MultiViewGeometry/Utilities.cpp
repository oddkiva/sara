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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>


namespace DO { namespace Sara {

  auto range(int n) -> Tensor_<int, 1>
  {
    auto indices = Tensor_<int, 1>{n};
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }

  auto random_samples(int num_samples,      //
                      int sample_size,      //
                      int num_data_points)  //
      -> Tensor_<int, 2>
  {
    auto indices = range(num_data_points);

    auto samples = Tensor_<int, 2>{{num_samples, sample_size}};
    for (int i = 0; i < num_samples; ++i)
    {
      auto indices_shuffled = shuffle(indices);
      samples[i].flat_array() = indices_shuffled.flat_array().head(sample_size);
    }

    return samples;
  }

} /* namespace Sara */
} /* namespace DO */
