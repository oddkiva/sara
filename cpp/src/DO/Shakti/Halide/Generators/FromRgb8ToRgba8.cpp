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

#include <DO/Shakti/Halide/MyHalide.hpp>
#include <Halide.h>


namespace {

  using namespace Halide;

  class Rgb8ToRgba8 : public Halide::Generator<Rgb8ToRgba8>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 8};

    Input<Buffer<std::uint8_t>> input{"Rgb8u", 3};
    Output<Buffer<std::uint8_t>> output{"Rgba8u", 3};

    Var x{"x"}, y{"y"}, c{"c"}, xi{"xi"}, yi{"yi"};

    void generate()
    {
      // Deal with interleaved RGB pixel format.
      input.dim(0).set_stride(3).dim(2).set_stride(1);
      input.dim(2).set_bounds(0, 3);

      output.dim(0).set_stride(4).dim(2).set_stride(1);
      output.dim(2).set_bounds(0, 4);

      auto input_ext = BoundaryConditions::constant_exterior(input, 255);
      output(x, y, c) = input_ext(x, y, c);

      output.reorder(c, x, y).unroll(c);

      schedule_algorithm();
    }

    void schedule_algorithm()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
        output.gpu_tile(x, y, xi, yi, tile_x, tile_y);

      // Hexagon schedule.
      else if (get_target().features_any_of({Halide::Target::HVX_v62,  //
                                             Halide::Target::HVX_v65,
                                             Halide::Target::HVX_v66,
                                             Halide::Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        output.hexagon()
            .prefetch(input, y, y, 2)
            .split(y, y, yi, 128)
            .parallel(y)
            .vectorize(x, vector_size);
      }

      // CPU schedule.
      else
        output.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(Rgb8ToRgba8, shakti_rgb8_to_rgba8_cpu)
HALIDE_REGISTER_GENERATOR(Rgb8ToRgba8, shakti_rgb8_to_rgba8_gpu)
