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

  struct SeparableConvolutionComponent
  {
    //! @brief Intermediate function
    //! @{
    Func conv_y_t{"conv_y_transposed"};
    Func conv_y{"conv_y"};
    //! @}

    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};
    //! @}

    template <typename Input, typename Output>
    void generate(Input& input, Func& kernel_x, Expr kernel_x_size,
                  Expr kernel_x_shift, Func& kernel_y, Expr kernel_y_size,
                  Expr kernel_y_shift, Output& output, int w, int h)
    {
      // const auto w = input.dim(0).extent();
      // const auto h = input.dim(1).extent();

      // Define the summation variable `k` with its summation domain (a.k.a. the
      // reduction domain variable).
      auto k = RDom{kernel_x_shift, kernel_x_size};
      auto l = RDom{kernel_y_shift, kernel_y_size};

      // 1st pass: transpose and convolve the columns.
      auto input_t = Func{"input_transposed"};
      input_t(x, y, c, n) = input(y, x, c, n);
      auto input_t_padded =
          BoundaryConditions::repeat_edge(input_t, {{0, h}, {}, {}, {}});
      conv_y_t(x, y, c, n) = sum(input_t_padded(x + l, y, c, n) * kernel_y(l));

      // 2nd pass: transpose and convolve the rows.
      auto conv_y = Func{"conv_y"};
      conv_y(x, y, c, n) = conv_y_t(y, x, c, n);
      auto conv_y_padded =
          BoundaryConditions::repeat_edge(conv_y, {{0, w}, {}, {}, {}});
      auto& conv_x = output;
      conv_x(x, y, c, n) = sum(conv_y_padded(x + k, y, c, n) * kernel_x(k));
    }

    template <typename Output>
    void schedule(const Halide::Target& target, int32_t tile_x, int32_t tile_y,
                  Output& output)
    {
      auto& conv_x = output;

      // GPU schedule.
      if (target.has_gpu_feature())
      {
        // 1st pass: transpose and convolve the columns
        conv_y_t.compute_root();
        conv_y_t.gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                          TailStrategy::GuardWithIf);

        // 2nd pass: transpose and convolve the rows.
        conv_x.gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                        TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (target.features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size = target.has_feature(Target::HVX_128) ? 128 : 64;

        // 1st pass: transpose and convolve the columns
        conv_y_t.compute_root();
        conv_y_t.hexagon()
            .prefetch(conv_y_t, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);

        // 2nd pass: transpose and convolve the rows.
        conv_y.compute_root();
        conv_x.hexagon()
            .prefetch(conv_y_t, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);
      }

      // CPU schedule.
      else
      {
        // 1st pass: transpose and convolve the columns
        conv_y_t.compute_root();
        conv_y_t.split(y, yo, yi, 8).parallel(yo).vectorize(x, 8);

        // 2nd pass: transpose and convolve the rows.
        conv_y.compute_root();
        conv_x.split(y, yo, yi, 8).parallel(yo).vectorize(x, 8);
      }
    }
  };

}  // namespace
