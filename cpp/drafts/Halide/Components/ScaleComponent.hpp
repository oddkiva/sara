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

  struct ScaleComponent
  {
    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};
    //! @}

    template <typename Input, typename Output>
    void generate(Input& input, Output& output, Expr w_in, Expr h_in,
                  Expr w_out, Expr h_out)
    {
      auto wi = cast<float>(w_in);
      auto hi = cast<float>(h_in);
      auto wo = cast<float>(w_out);
      auto ho = cast<float>(h_out);

      auto xx = cast<int32_t>(x * wi / wo);
      auto yy = cast<int32_t>(y * hi / ho);

      output(x, y, c, n) = input(xx, yy, c, n);
    }

    template <typename Input, typename Output>
    void schedule(const Halide::Target& target, int32_t tile_x, int32_t tile_y,
                  Input& input, Output& output)
    {
      output.dim(0).set_stride(Expr());

      Expr output_is_planar = output.dim(0).stride() == 1;
      Expr output_is_interleaved = output.dim(0).stride() == 3 &&  //
                                   output.dim(2).stride() == 1 &&  //
                                   output.dim(2).extent() == 3;

      // GPU schedule.
      if (target.has_gpu_feature())
      {
        output.specialize(output_is_planar)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                      TailStrategy::GuardWithIf);

        output.specialize(output_is_interleaved)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 3,
                      TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (target.features_any_of({Halide::Target::HVX_v62,  //
                                       Halide::Target::HVX_v65,
                                       Halide::Target::HVX_v66,
                                       Halide::Target::HVX_128}))
      {
        const auto vector_size = target.has_feature(Target::HVX_128) ? 128 : 64;

        output.specialize(output_is_planar)
            .hexagon()
            .prefetch(input, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        output.specialize(output_is_planar)
            .split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);

        output.specialize(output_is_interleaved).reorder(c, x, y).unroll(c);
      }
    }
  };

}  // namespace
