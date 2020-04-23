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

  template <typename T>
  class ConvolutionGenerator : public Generator<ConvolutionGenerator<T>>
  {
  public:
    using Base = Generator<GemmBasedConvolutionGenerator<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Buffer<T>> input{"im2col", 4};

    Input<Buffer<T>> kernel{"kernel", 4};
    Input<int[3]> kernel_begin{"k_begin"};
    Input<int[3]> kernel_extent{"k_extent"};

    Output<Buffer<T>> output{"output", 4};

    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"}, c{"s"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, co{"so"};
    Var xi{"xi"}, yi{"yi"}, ci{"si"};
    //! @}

    void generate()
    {
      const auto x0 = kernel_begin[0];
      const auto y0 = kernel_begin[1];
      const auto c0 = kernel_begin[2];

      const auto kw = kernel_extent[0];
      const auto kh = kernel_extent[1];
      const auto kc = kernel_extent[2];

      auto r = RDom{x0, kw, y0, kh, c0, kc};

      output(x, y, c, n) =
          sum(input(x + r.x, y + r.y, c + r.z, n) * kernel(r.x, r.y, r.z, c));
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        output.gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
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

HALIDE_REGISTER_GENERATOR(ConvolutionGenerator<float>, convolve_32f)
HALIDE_REGISTER_GENERATOR(ConvolutionGenerator<double>, convolve_64f)
