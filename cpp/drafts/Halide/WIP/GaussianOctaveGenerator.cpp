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

#include <DO/Sara/ImageProcessing/GaussianPyramid.hpp>


namespace {

  using namespace Halide;

  class GaussianOctaveGenerator : public Generator<GaussianOctaveGenerator>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Func> im2col{"input", Float(32), 2};

    Input<Func> kernels{"kernel", Float(32), 3};
    Input<int32_t[2]> kernel_begin{"kernel_begin"};
    Input<int32_t[2]> kernel_extent{"kernel_size"};

    Output<Func> output{"output", Float(32), 3};

    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"}, s{"s"};
    Var xo{"xo"}, yo{"yo"}, so{"so"};
    Var xi{"xi"}, yi{"yi"}, si{"si"};
    //! @}

    void generate()
    {
      const auto x0 = kernel_begin[0];
      const auto y0 = kernel_begin[1];

      const auto kw = kernel_extent[0];
      const auto kh = kernel_extent[1];

      auto r = RDom{x0, kw, y0, kh};

      output(x, y, s) = sum(im2col(x + r.x, y + r.y) * kernel(r.x, r.y, s));
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        output.gpu_tile(x, y, s, xo, yo, so, xi, yi, si, tile_x, tile_y, 1,
                        TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        output.hexagon()
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        output.split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);
      }
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(GaussianOctaveGenerator, halide_gaussian_octave_generator)
