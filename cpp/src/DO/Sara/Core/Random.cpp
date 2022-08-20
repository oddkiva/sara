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
#include <DO/Sara/Core/Numpy.hpp>
#include <DO/Sara/Core/Random.hpp>


namespace DO::Sara {

  auto random_samples(int num_samples,      //
                      int sample_size,      //
                      int num_data_points)  //
      -> Tensor_<int, 2>
  {
    auto rd = std::random_device{};
    auto g = std::mt19937{rd()};

    auto indices = range(num_data_points);

    auto samples = Tensor_<int, 2>{{num_samples, sample_size}};
    for (int i = 0; i < num_samples; ++i)
    {
      auto samples_i = samples[i];
      std::sample(indices.begin(), indices.end(), samples_i.begin(),
                  sample_size, g);
    }

    return samples;
  }

} /* namespace DO::Sara */
