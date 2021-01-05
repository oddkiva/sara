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

#include <drafts/Halide/MyHalide.hpp>


namespace {

  using namespace Halide;

  class RgbToGray : public Halide::Generator<RgbToGray>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 8};

    Input<Buffer<std::uint8_t>> input{"Rgb", 3};
    Output<Buffer<float>> output{"Gray", 2};

    Var x{"x"}, y{"y"}, xi{"xi"}, yi{"yi"};

    void generate()
    {
      // Deal with interleaved RGB pixel format.
      input.dim(0).set_stride(3).dim(2).set_stride(1);
      input.dim(2).set_bounds(0, 3);

      auto r = input(x, y, 0) / 255.f;
      auto g = input(x, y, 1) / 255.f;
      auto b = input(x, y, 2) / 255.f;

      auto gray = Func{"gray"};
      output(x, y) = 0.2125f * r + 0.7154f * g + 0.0721f * b;

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

}  // namespace


HALIDE_REGISTER_GENERATOR(RgbToGray, shakti_halide_rgb_to_gray)
