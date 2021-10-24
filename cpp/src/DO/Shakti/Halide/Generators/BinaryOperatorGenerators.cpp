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


  struct Minus
  {
    inline Expr operator()(const Expr& a, const Expr& b) const
    {
      return a - b;
    };
  };


  template <typename T, typename Op>
  class BinaryOp : public Generator<BinaryOp<T, Op>>
  {
    using Base = Generator<BinaryOp<T, Op>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

  public:
    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};

    Input<Buffer<T>> a{"a", 4};
    Input<Buffer<T>> b{"b", 4};
    Op op;

    Output<Buffer<T>> out{"out", 4};

    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};
    //! @}

    void generate()
    {
      out(x, y, c, n) = op(a(x, y, c, n), b(x, y, c, n));
    }

    void schedule()
    {
      a.dim(0).set_stride(Expr());
      b.dim(0).set_stride(Expr());
      out.dim(0).set_stride(Expr());

      Expr a_is_planar = a.dim(0).stride() == 1;
      Expr a_is_interleaved = a.dim(0).stride() == 3 &&  //
                              a.dim(2).stride() == 1 &&  //
                              a.dim(2).extent() == 3;

      Expr b_is_planar = b.dim(0).stride() == 1;
      Expr b_is_interleaved = b.dim(0).stride() == 3 &&  //
                              b.dim(2).stride() == 1 &&  //
                              b.dim(2).extent() == 3;

      Expr out_is_planar = out.dim(0).stride() == 1;
      Expr out_is_interleaved = out.dim(0).stride() == 3 &&  //
                                out.dim(2).stride() == 1 &&  //
                                out.dim(2).extent() == 3;

      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        out.specialize(a_is_planar && out_is_planar)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                      TailStrategy::GuardWithIf);

        out.specialize(a_is_interleaved && out_is_interleaved)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 3,
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

        out.specialize(a_is_planar && out_is_planar)
            .hexagon()
            .prefetch(a, y, 2)
            .prefetch(b, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        out.specialize(a_is_planar && b_is_planar && out_is_planar)
            .split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);

        out.specialize(a_is_interleaved && b_is_interleaved &&
                       out_is_interleaved)
            .reorder(c, x, y)
            .unroll(c);
      }
    }
  };

  template <typename T>
  using Subtract = BinaryOp<T, Minus>;

}  // namespace

HALIDE_REGISTER_GENERATOR(Subtract<float>, shakti_subtract_32f_cpu)
HALIDE_REGISTER_GENERATOR(Subtract<float>, shakti_subtract_32f_gpu)
