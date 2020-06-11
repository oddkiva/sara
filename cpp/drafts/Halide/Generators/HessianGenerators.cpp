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
  class Hessian : public Generator<Hessian<T>>
  {
  public:
    using Base = Generator<Hessian<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};

    Input<Buffer<T>> prev{"prev", 4};
    Input<Buffer<T>> curr{"curr", 4};
    Input<Buffer<T>> next{"next", 4};

    Output<Buffer<T>> output{"output", 4};

    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, co{"co"}, no{"no"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"}, ni{"ni"};

    void generate()
    {
      // Gradient
      Expr dx = curr(x + 1, y, c, n) - curr(x - 1, y, c, n);
      Expr dy = curr(x, y + 1, c, n) - curr(x, y - 1, c, n);
      Expr ds = next(x, y, c, n) - prev(x, y, c, n);

      // Hessian
      Expr dxx =
          curr(x + 1, y, c, n) + curr(x - 1, y, c, n) - 2 * curr(x, y, c, n);
      Expr dyy =
          curr(x, y + 1, c, n) + curr(x, y - 1, c, n) - 2 * curr(x, y, c, n);
      Expr dss = next(x, y, c, n) + prev(x, y - 1, c, n) - 2 * curr(x, y, c, n);
      Expr dxy = (curr(x + 1, y + 1, c, n) - curr(x - 1, y - 1, c, n) -
                  curr(x + 1, y - 1, c, n) + curr(x - 1, y - 1, c, n)) /
                 4;
      Expr dxs = (next(x + 1, y, c, n) - next(x - 1, y, c, n) -
                  prev(x + 1, y, c, n) + prev(x - 1, y, c, n)) /
                 4;
      Expr dys = (next(x, y + 1, c, n) - next(x, y - 1, c, n) -
                  prev(x, y + 1, c, n) + prev(x, y - 1, c, n)) /
                 4;

      //
    }

    void schedule()
    {
      // input.dim(0).set_stride(Expr());  // Use an undefined Expr to
      //                                   // mean there is no
      //                                   // constraint.

      // output.dim(0).set_stride(Expr());

      // Expr input_is_planar = input.dim(0).stride() == 1;
      // Expr input_is_interleaved = input.dim(0).stride() == 3 &&  //
      //                             input.dim(2).stride() == 1 &&  //
      //                             input.dim(2).extent() == 3;

      // Expr output_is_planar = output.dim(0).stride() == 1;
      // Expr output_is_interleaved = output.dim(0).stride() == 3 &&  //
      //                              output.dim(2).stride() == 1 &&  //
      //                              output.dim(2).extent() == 3;

      // // GPU schedule.
      // if (get_target().has_gpu_feature())
      // {
      //   output.specialize(input_is_planar && output_is_planar)
      //       .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
      //                 TailStrategy::GuardWithIf);

      //   output.specialize(input_is_interleaved && output_is_interleaved)
      //       .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 3,
      //                 TailStrategy::GuardWithIf);
      // }

      // // Hexagon schedule.
      // else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      // {
      //   const auto vector_size =
      //       get_target().has_feature(Target::HVX_128) ? 128 : 64;

      //   output.specialize(input_is_planar && output_is_planar)
      //       .hexagon()
      //       .prefetch(input, y, 2)
      //       .split(y, yo, yi, 128)
      //       .parallel(yo)
      //       .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      // }

      // // CPU schedule.
      // else
      // {
      //   output.specialize(input_is_planar && output_is_planar)
      //       .split(y, yo, yi, 8)
      //       .parallel(yo)
      //       .vectorize(x, 8, TailStrategy::GuardWithIf);

      //   output.specialize(input_is_interleaved && output_is_interleaved)
      //       .reorder(c, x, y)
      //       .unroll(c);
      // }
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(Hessian<float>, shakti_hessian_32f)
