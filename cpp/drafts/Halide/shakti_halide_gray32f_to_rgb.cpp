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


#include "MyHalide.hpp"


using namespace Halide;


class Gray32fToRgb : public Halide::Generator<Gray32fToRgb>
{
public:
  GeneratorParam<int> tile_x{"tile_x", 32};
  GeneratorParam<int> tile_y{"tile_y", 8};

  Input<Buffer<float>> input{"gray32f", 2};
  Output<Buffer<std::uint8_t>> output{"packed_rgb888", 3};

  // Target target{get_gpu_target()};

  Var x{"x"}, y{"y"}, c{"c"}, xi{"xi"}, yi{"yi"};

  void generate()
  {
    output(x, y, c) = cast<uint8_t>(input(x, y) * 255.f);

    output.dim(0).set_stride(3).dim(2).set_stride(1);
    output.dim(2).set_bounds(0, 3);
    output.reorder(c, x, y).unroll(c);

    // GPU schedule.
    if (get_target().has_gpu_feature())
      output.gpu_tile(x, y, xi, yi, tile_x, tile_y);

    // Hexagon schedule.
    else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
    {
      const auto vector_size =
          get_target().has_feature(Target::HVX_128) ? 128 : 64;

      output.hexagon()
          .prefetch(input, y, 2)
          .split(y, y, yi, 128)
          .parallel(y)
          .vectorize(x, vector_size);
    }

    // CPU schedule.
    else
      output.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
  }
};

HALIDE_REGISTER_GENERATOR(Gray32fToRgb, shakti_halide_gray32f_to_rgb)
