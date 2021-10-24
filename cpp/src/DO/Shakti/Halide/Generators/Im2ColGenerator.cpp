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


namespace {

  using namespace Halide;

  template <typename T>
  class Im2ColGenerator : public Generator<Im2ColGenerator<T>>
  {
  public:
    using Base = Generator<Im2ColGenerator<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;


    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Buffer<T>> input{"input", 4};
    Input<int[3]> k_begin{"k_begin"};
    Input<int[3]> k_extent{"k_extent"};

    Output<Buffer<T>> output{"output", 3};

    //! @brief Variables.
    //! @{
    Var i{"i"}, j{"j"}, n{"n"};
    Var io{"xo"}, jo{"yo"}, no{"no"};
    Var ii{"xi"}, ji{"yi"}, ni{"ni"};
    //! @}

    void generate()
    {
      auto input_padded = BoundaryConditions::constant_exterior(input, 0);

      // In principle this is what we should do:
      // output(i, j, c, x, y, n) = input(x + i, y + j, c, n);
      //
      // The GPU hardware is optimized for arrays up to dimension 3.

      const auto w = input.dim(0).extent();

      const auto x0 = k_begin[0];
      const auto y0 = k_begin[1];
      const auto c0 = k_begin[2];

      const auto kw = k_extent[0];
      const auto kh = k_extent[1];
      const auto kc = k_extent[2];

      // j = w * y + x;
      Expr y = j / w;
      Expr x = j % w;

      // i = kw * kh * (-c0 + i.c) + kw * (-y0 + i.y) + (-x0 + i.x)
      //               -----------   ----------------   ----------
      //                    |                  |             |
      //     kw * kw *      ci     + kw *      yi     +      xi
      Expr ci = i / (kw * kh) + c0;
      Expr yi = (i - kw * kh * (ci - c0)) / kw + y0;
      Expr xi = i % kw + x0;

      output(i, j, n) = input_padded(x + xi, y + yi, ci, n);
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        output.gpu_tile(i, j, n, io, jo, no, ii, ji, ni, tile_x, tile_y, 1,
                        TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Halide::Target::HVX_v62,  //
                                             Halide::Target::HVX_v65,
                                             Halide::Target::HVX_v66,
                                             Halide::Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        output.hexagon()
            .split(j, jo, ji, 128)
            .parallel(jo)
            .vectorize(i, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        output.split(j, jo, ji, 8)
            .parallel(jo)
            .vectorize(i, 8, TailStrategy::GuardWithIf);
      }
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(Im2ColGenerator<float>, shakti_im2col_32f_gpu)
HALIDE_REGISTER_GENERATOR(Im2ColGenerator<double>, shakti_im2col_64f_gpu)
