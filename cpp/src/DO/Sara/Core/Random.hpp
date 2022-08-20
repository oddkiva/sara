// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Tensor.hpp>

#include <random>


namespace DO::Sara {

  template <typename T, typename RandomGenerator>
  inline auto shuffle(const TensorView_<T, 1>& x, RandomGenerator& g)
      -> Tensor_<T, 1>
  {
    auto x_shuffled = x;
    std::shuffle(x_shuffled.begin(), x_shuffled.end(), g);
    return x_shuffled;
  }

  template <typename T, typename RandomGenerator>
  inline auto sample(const TensorView_<T, 1>& x, const int sample_size,
                     RandomGenerator& g) -> Tensor_<T, 1>
  {
    auto x_sampled = Tensor_<T, 1>{sample_size};
    std::sample(x.begin(), x.end(), sample_size, g);
    return x_sampled;
  }

  DO_SARA_EXPORT
  auto random_samples(int num_samples,      //
                      int sample_size,      //
                      int num_data_points)  //
      -> Tensor_<int, 2>;

}  // namespace DO::Sara
