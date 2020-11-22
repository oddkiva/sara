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


namespace DO { namespace Shakti { namespace HalideBackend {

  struct SeparableConvolution
  {
    //! @brief Intermediate function
    //! @{
    Halide::Func conv_y_t{"conv_y_transposed"};
    Halide::Func conv_y{"conv_y"};
    //! @}

    //! @brief Variables.
    //! @{
    Halide::Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Halide::Var xo{"xo"}, yo{"yo"}, co{"co"};
    Halide::Var xi{"xi"}, yi{"yi"}, ci{"ci"};
    //! @}

    template <typename Input, typename Output>
    void generate(Input& input,                                             //
                  Halide::Func& kernel_x,                                   //
                  Halide::Expr kernel_x_size, Halide::Expr kernel_x_shift,  //
                  Halide::Func& kernel_y,                                   //
                  Halide::Expr kernel_y_size, Halide::Expr kernel_y_shift,  //
                  Output& output,                                           //
                  Halide::Expr w, Halide::Expr h)
    {
      // Define the summation variable `k` with its summation domain (a.k.a. the
      // reduction domain variable).
      auto k = Halide::RDom{kernel_x_shift, kernel_x_size};
      auto l = Halide::RDom{kernel_y_shift, kernel_y_size};

      // 1st pass: transpose and convolve the columns.
      auto input_t = Halide::Func{"input_transposed"};
      input_t(x, y, c, n) = input(y, x, c, n);
      auto input_t_padded = Halide::BoundaryConditions::repeat_edge(
          input_t, {{0, h}, {}, {}, {}});
      conv_y_t(x, y, c, n) = sum(input_t_padded(x + l, y, c, n) * kernel_y(l));

      // 2nd pass: transpose and convolve the rows.
      auto conv_y = Halide::Func{"conv_y"};
      conv_y(x, y, c, n) = conv_y_t(y, x, c, n);
      auto conv_y_padded =
          Halide::BoundaryConditions::repeat_edge(conv_y, {{0, w}, {}, {}, {}});
      auto& conv_x = output;
      conv_x(x, y, c, n) =
          Halide::sum(conv_y_padded(x + k, y, c, n) * kernel_x(k));
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
                          Halide::TailStrategy::GuardWithIf);

        // 2nd pass: transpose and convolve the rows.
        conv_x.gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                        Halide::TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (target.features_any_of(
                   {Halide::Target::HVX_64, Halide::Target::HVX_128}))
      {
        const auto vector_size =
            target.has_feature(Halide::Target::HVX_128) ? 128 : 64;

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

    template <typename Input, typename Output>
    void
    generate_2d(Input& input,                                             //
                Halide::Func& kernel_x,                                   //
                Halide::Expr kernel_x_size, Halide::Expr kernel_x_shift,  //
                Halide::Func& kernel_y,                                   //
                Halide::Expr kernel_y_size, Halide::Expr kernel_y_shift,  //
                Output& output,                                           //
                Halide::Expr w, Halide::Expr h)                           //
    {
      // Define the summation variable `k` with its summation domain (a.k.a. the
      // reduction domain variable).
      auto k = Halide::RDom{kernel_x_shift, kernel_x_size};
      auto l = Halide::RDom{kernel_y_shift, kernel_y_size};

      // 1st pass: transpose and convolve the columns.
      auto input_t = Halide::Func{"input_transposed"};
      input_t(x, y) = input(y, x);
      auto input_t_padded =
          Halide::BoundaryConditions::repeat_edge(input_t, {{0, h}, {}});
      conv_y_t(x, y) = sum(input_t_padded(x + l, y) * kernel_y(l));

      // 2nd pass: transpose and convolve the rows.
      auto conv_y = Halide::Func{"conv_y"};
      conv_y(x, y) = conv_y_t(y, x);
      auto conv_y_padded =
          Halide::BoundaryConditions::repeat_edge(conv_y, {{0, w}, {}});
      auto& conv_x = output;
      conv_x(x, y) = Halide::sum(conv_y_padded(x + k, y) * kernel_x(k));
    }

    template <typename Output>
    void schedule_2d(const Halide::Target& target, int32_t tile_x,
                     int32_t tile_y, Output& output)
    {
      auto& conv_x = output;

      // GPU schedule.
      if (target.has_gpu_feature())
      {
        // 1st pass: transpose and convolve the columns
        conv_y_t.compute_root();
        conv_y_t.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y,
                          Halide::TailStrategy::GuardWithIf);

        // 2nd pass: transpose and convolve the rows.
        conv_x.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y,
                        Halide::TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (target.features_any_of(
                   {Halide::Target::HVX_64, Halide::Target::HVX_128}))
      {
        const auto vector_size =
            target.has_feature(Halide::Target::HVX_128) ? 128 : 64;

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
        conv_y_t.split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8, Halide::TailStrategy::GuardWithIf);

        // 2nd pass: transpose and convolve the rows.
        conv_y.compute_root();
        conv_x.split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8, Halide::TailStrategy::GuardWithIf);
      }
    }
  };

}}}  // namespace DO::Shakti::HalideBackend
