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
  class LocalMax : public Generator<LocalMax<T>>
  {
    using Base = Generator<LocalMax<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

  public:
    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};

    Input<Buffer<T>> scale_prev{"prev", 4};
    Input<Buffer<T>> scale_curr{"curr", 4};
    Input<Buffer<T>> scale_next{"next", 4};

    //Output<Buffer<std::uint8_t>> out{"out", 4};
    Output<Buffer<int32_t>> out{"out", 4};

    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};
    //! @}

    void generate()
    {
      auto a_ext = BoundaryConditions::repeat_edge(scale_prev);
      auto b_ext = BoundaryConditions::repeat_edge(scale_curr);
      auto c_ext = BoundaryConditions::repeat_edge(scale_next);

      out(x, y, c, n) =
          select(b_ext(x, y, c, n) > a_ext(x - 1, y - 1, c, n) &&
                 b_ext(x, y, c, n) > a_ext(x + 0, y - 1, c, n) &&
                 b_ext(x, y, c, n) > a_ext(x + 1, y - 1, c, n) &&
                 b_ext(x, y, c, n) > a_ext(x - 1, y + 0, c, n) &&
                 b_ext(x, y, c, n) > a_ext(x + 0, y + 0, c, n) &&
                 b_ext(x, y, c, n) > a_ext(x + 1, y + 0, c, n) &&
                 b_ext(x, y, c, n) > a_ext(x - 1, y + 1, c, n) &&
                 b_ext(x, y, c, n) > a_ext(x + 0, y + 1, c, n) &&
                 b_ext(x, y, c, n) > a_ext(x + 1, y + 1, c, n) &&
                 //
                 b_ext(x, y, c, n) > b_ext(x - 1, y - 1, c, n) &&
                 b_ext(x, y, c, n) > b_ext(x + 0, y - 1, c, n) &&
                 b_ext(x, y, c, n) > b_ext(x + 1, y - 1, c, n) &&
                 b_ext(x, y, c, n) > b_ext(x - 1, y + 0, c, n) &&
                 // b_ext(x, y, c, n) > b_ext(x + 0, y + 0, c, n) &&
                 b_ext(x, y, c, n) > b_ext(x + 1, y + 0, c, n) &&
                 b_ext(x, y, c, n) > b_ext(x - 1, y + 1, c, n) &&
                 b_ext(x, y, c, n) > b_ext(x + 0, y + 1, c, n) &&
                 b_ext(x, y, c, n) > b_ext(x + 1, y + 1, c, n) &&
                 //
                 b_ext(x, y, c, n) > c_ext(x - 1, y - 1, c, n) &&
                 b_ext(x, y, c, n) > c_ext(x + 0, y - 1, c, n) &&
                 b_ext(x, y, c, n) > c_ext(x + 1, y - 1, c, n) &&
                 b_ext(x, y, c, n) > c_ext(x - 1, y + 0, c, n) &&
                 b_ext(x, y, c, n) > c_ext(x + 0, y + 0, c, n) &&
                 b_ext(x, y, c, n) > c_ext(x + 1, y + 0, c, n) &&
                 b_ext(x, y, c, n) > c_ext(x - 1, y + 1, c, n) &&
                 b_ext(x, y, c, n) > c_ext(x + 0, y + 1, c, n) &&
                 b_ext(x, y, c, n) > c_ext(x + 1, y + 1, c, n),
                 1, 0);
    }

    void schedule()
    {
      auto& prev = scale_prev;
      auto& curr = scale_curr;
      auto& next = scale_next;

      prev.dim(0).set_stride(Expr());
      curr.dim(0).set_stride(Expr());
      next.dim(0).set_stride(Expr());
      out.dim(0).set_stride(Expr());

      Expr prev_is_planar = prev.dim(0).stride() == 1;
      Expr prev_is_interleaved = prev.dim(0).stride() == 3 &&  //
                                 prev.dim(2).stride() == 1 &&  //
                                 prev.dim(2).extent() == 3;

      Expr curr_is_planar = curr.dim(0).stride() == 1;
      Expr curr_is_interleaved = curr.dim(0).stride() == 3 &&  //
                                 curr.dim(2).stride() == 1 &&  //
                                 curr.dim(2).extent() == 3;

      Expr next_is_planar = next.dim(0).stride() == 1;
      Expr next_is_interleaved = next.dim(0).stride() == 3 &&  //
                                 next.dim(2).stride() == 1 &&  //
                                 next.dim(2).extent() == 3;

      Expr out_is_planar = out.dim(0).stride() == 1;
      Expr out_is_interleaved = out.dim(0).stride() == 3 &&  //
                                out.dim(2).stride() == 1 &&  //
                                out.dim(2).extent() == 3;

      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        out.specialize(prev_is_planar && curr_is_planar && next_is_planar &&
                       out_is_planar)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                      TailStrategy::GuardWithIf);

        out.specialize(prev_is_interleaved && curr_is_interleaved &&
                       next_is_interleaved && out_is_interleaved)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 3,
                      TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        out.specialize(prev_is_planar && curr_is_planar && next_is_planar &&
                       out_is_planar)
            .hexagon()
            .prefetch(prev, y, 2)
            .prefetch(curr, y, 2)
            .prefetch(next, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        out.specialize(prev_is_planar && curr_is_planar && next_is_planar &&
                       out_is_planar)
            .split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);

        out.specialize(prev_is_interleaved && curr_is_interleaved &&
                       next_is_interleaved && out_is_interleaved)
            .reorder(c, x, y)
            .unroll(c);
      }
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(LocalMax<float>, shakti_local_max_32f)
