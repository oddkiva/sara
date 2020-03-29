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

#include <Halide.h>


namespace {

  using namespace Halide;

  class CastFromUint8ToFloat : public Generator<CastFromUint8ToFloat>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 8};
    GeneratorParam<int> tile_y{"tile_y", 8};

    Input<Buffer<std::uint8_t>> input{"input", 3};
    Output<Buffer<float>> output{"output", 3};

    Var x{"x"}, y{"y"}, c{"c"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};

    void generate()
    {
      output(x, y, c) = input(x, y, c) / 255.f;
    }

    void schedule()
    {
      input.dim(0).set_stride(Expr());
      output.dim(0).set_stride(Expr());

      Expr input_is_planar = input.dim(0).stride() == 1;
      Expr input_is_interleaved = input.dim(0).stride() == 3 and  //
                                  input.dim(2).stride() == 1 and  //
                                  input.dim(2).extent() == 3;

      Expr output_is_planar = output.dim(0).stride() == 1;
      Expr output_is_interleaved = output.dim(0).stride() == 3 and  //
                                   output.dim(2).stride() == 1 and  //
                                   output.dim(2).extent() == 3;

      // GPU schedule.
      if (get_target().has_gpu_feature())
        output.gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 3);

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        output.specialize(input_is_planar && output_is_planar)
            .hexagon()
            .prefetch(input, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);
      }

      // CPU schedule.
      else
      {
        output.specialize(input_is_planar && output_is_planar)
            .split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8);

        output.specialize(input_is_interleaved && output_is_interleaved)
            .reorder(c, x, y)
            .unroll(c);
      }
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(CastFromUint8ToFloat, shakti_halide_cast_to_float,
                          HalideBackend::CastFromUint8ToFloat)
