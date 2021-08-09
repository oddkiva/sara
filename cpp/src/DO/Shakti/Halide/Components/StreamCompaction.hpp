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

#pragma once

#include <DO/Shakti/Halide/MyHalide.hpp>


namespace DO::Shakti::HalideBackend {

  template <typename Input>
  auto flatten_4d(const Input& f, const Halide::Var& x,  //
                  std::int32_t w, std::int32_t h, std::int32_t c)
  {
    auto f_flattened = Halide::Func{};
    f_flattened(x) = f(x % w, (x / w) % h, (x / (w * h)) % c, x / (w * h * c));
    return f_flattened;
  }

  template <typename Input>
  auto unflatten_4d(const Input& indices_flattened, const Halide::Var& x,  //
                    std::int32_t w, std::int32_t h, std::int32_t c)
  {
    auto x_indices = Halide::Func{"x"};
    auto y_indices = Halide::Func{"y"};
    auto c_indices = Halide::Func{"c"};
    auto n_indices = Halide::Func{"n"};

    const auto flat_index = indices_flattened(x);
    x_indices(x) = select(flat_index != -1, flat_index % w, -1);
    y_indices(x) = select(flat_index != -1, (flat_index / w) % h, -1);
    c_indices(x) = select(flat_index != -1, (flat_index / (w * h)) % c, -1);
    n_indices(x) = select(flat_index != -1, flat_index / (w * h * c), -1);

    return std::make_tuple(x_indices, y_indices, c_indices, n_indices);
  }

  auto non_zeros_indices(Halide::Func& indices, const Halide::Var& x,
                         std::int32_t size)
  {
    using namespace Halide;

  // Express the prefix sum by a recurrence relation.
    Func prefix_sum;
    prefix_sum(x) = cast<int32_t>(0);

    RDom range(1, size - 1);
    prefix_sum(range) = select(indices(range - 1) != 0,    //
                               prefix_sum(range - 1) + 1,  //
                               prefix_sum(range - 1));

    // Compacted indices.
    RDom in_range(0, size);
    Func compacted_indices;
    compacted_indices(x) = -1;
    compacted_indices(clamp(prefix_sum(in_range), 0, size - 1)) =
        select(indices(in_range) != 0, in_range, -1);

    return compacted_indices;
  }


}  // namespace DO::Shakti::HalideBackend
