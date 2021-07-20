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


namespace DO::Shakti::HalideBackend {

  struct SeparableConvolution
  {
    //! @brief Intermediate function
    //! @{
    Halide::Func conv_x{"conv_x"};
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

      // 1st pass: convolve in x first.
      auto input_padded = Halide::BoundaryConditions::repeat_edge(  //
          input,                                                    //
          {{0, w}, {}, {}, {}}                                      //
      );
      conv_x(x, y, c, n) = sum(input_padded(x + k, y, c, n) * kernel_x(k));

      // 2nd pass: transpose and convolve the rows.
      auto conv_x_padded = Halide::BoundaryConditions::repeat_edge(  //
          conv_x,                                                    //
          {{}, {0, h}, {}, {}}                                       //
      );

      auto& conv_y = output;
      conv_y(x, y, c, n) =
          Halide::sum(conv_x_padded(x, y + l, c, n) * kernel_y(l));
    }

    template <typename Output>
    void schedule(const Halide::Target& target, int32_t tile_x, int32_t tile_y,
                  Output& output)
    {
      auto& conv_y = output;

      // GPU schedule.
      if (target.has_gpu_feature())
      {
        // 1st pass: transpose and convolve the columns
        conv_x.compute_root();
        conv_x.gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                        Halide::TailStrategy::GuardWithIf);

        // 2nd pass: transpose and convolve the rows.
        conv_y.gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                        Halide::TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (target.features_any_of(
                   {Halide::Target::HVX_v62, Halide::Target::HVX_v65,
                    Halide::Target::HVX_v66, Halide::Target::HVX_128}))
      {
        const auto vector_size =
            target.has_feature(Halide::Target::HVX_128) ? 128 : 64;

        // 1st pass: transpose and convolve the columns
        conv_x.compute_root();
        conv_x.hexagon()
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);

        // 2nd pass: transpose and convolve the rows.
        conv_y.hexagon()
            .prefetch(conv_x, x, 128)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);
      }

      // CPU schedule.
      else
      {
        // 1st pass.
        conv_x.compute_root();
        conv_x.split(y, yo, yi, 32).parallel(yo).vectorize(x, 8);

        // 2nd pass.
        conv_y.split(x, xo, xi, 32).parallel(xo).vectorize(y, 8);
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
      auto input_padded = Halide::BoundaryConditions::repeat_edge(  //
          input,                                                    //
          {{0, w}, {}}                                              //
      );
      conv_x(x, y) = sum(input_padded(x + k, y) * kernel_x(k));

      // 2nd pass: transpose and convolve the rows.
      auto conv_x_padded = Halide::BoundaryConditions::repeat_edge(  //
          conv_x,                                                    //
          {{}, {0, h}}                                               //
      );
      auto& conv_y = output;
      conv_y(x, y) = Halide::sum(conv_x_padded(x, y + l) * kernel_y(l));
    }

    template <typename Output>
    void schedule_2d(const Halide::Target& target, int32_t tile_x,
                     int32_t tile_y, Output& output)
    {
      auto& conv_y = output;

      // GPU schedule.
      if (target.has_gpu_feature())
      {
        // 1st pass: transpose and convolve the columns
        conv_x.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y,
                        Halide::TailStrategy::GuardWithIf);

        // 2nd pass: transpose and convolve the rows.
        conv_y.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y,
                        Halide::TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (target.features_any_of(
                   {Halide::Target::HVX_v62, Halide::Target::HVX_v65,
                    Halide::Target::HVX_v66, Halide::Target::HVX_128}))
      {
        const auto vector_size =
            target.has_feature(Halide::Target::HVX_128) ? 128 : 64;

        // 1st pass: transpose and convolve the columns
        conv_x.compute_root();
        conv_x.hexagon()
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);

        // 2nd pass: transpose and convolve the rows.
        conv_y.hexagon()
            .prefetch(conv_x, x, 128)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);
      }

      // CPU schedule.
      else
      {
        // 2nd pass: convolve the rows.
        conv_y.split(y, yo, yi, 32).parallel(yo).vectorize(x, 8);

        // Inside the 2nd pass, schedule the 1st pass.
        conv_x  //
            .store_at(conv_y, yo)
            .compute_at(conv_y, yi)
            .vectorize(x, 8);
      }
    }
  };

}  // namespace DO::Shakti::HalideBackend
