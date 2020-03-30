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

#include <Halide.h>


namespace {

  using namespace Halide;

  class SeparableConvolution2d
    : public Halide::Generator<SeparableConvolution2d>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Func> input{"input", Float(32), 3};
    Input<Func> kernel{"kernel", Float(32), 1};
    Input<int32_t> kernel_size{"kernel_size"};
    Input<int32_t> kernel_shift{"kernel_shift"};

    //! @brief Intermediate function
    //! @{
    Func conv_y_t{"conv_y_transposed"};
    Func conv_y{"conv_y"};
    //! @}

    Output<Func> output{"input_convolved", Float(32), 3};

    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"}, c{"c"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};
    //! @}

    void generate()
    {
      // Define the summation variable `k` with its summation domain (a.k.a. the
      // reduction domain variable).
      auto k = RDom{kernel_shift, kernel_size};

      // 1st pass: transpose and convolve the columns.
      auto input_t = Func{"input_transposed"};
      input_t(x, y, c) = input(y, x, c);
      conv_y_t(x, y, c) = sum(input_t(x + k, y, c) * kernel(k));

      // 2nd pass: transpose and convolve the rows.
      auto conv_y = Func{"conv_y"};
      conv_y(x, y, c) = conv_y_t(y, x, c);

      auto& conv_x = output;
      conv_x(x, y, c) = sum(conv_y(x + k, y, c) * kernel(k));
    }

    void schedule()
    {
      auto& conv_x = output;

      //// GPU schedule.
      //if (get_target().has_gpu_feature())
      //{
      //  // 1st pass: transpose and convolve the columns
      //  conv_y_t.compute_root();
      //  conv_y_t.gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1);

      //  // 2nd pass: transpose and convolve the rows.
      //  conv_x.gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1);
      //}

      //// Hexagon schedule.
      //else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      //{
      //  const auto vector_size =
      //      get_target().has_feature(Target::HVX_128) ? 128 : 64;

      //  // 1st pass: transpose and convolve the columns
      //  conv_y_t.compute_root();
      //  conv_y_t.hexagon()
      //      .prefetch(conv_y_t, y, 2)
      //      .split(y, yo, yi, 128)
      //      .parallel(yo)
      //      .vectorize(x, vector_size);

      //  // 2nd pass: transpose and convolve the rows.
      //  conv_y.compute_root();
      //  conv_x.hexagon()
      //      .prefetch(conv_y_t, y, 2)
      //      .split(y, yo, yi, 128)
      //      .parallel(yo)
      //      .vectorize(x, vector_size);
      //}

      //// CPU schedule.
      //else
      //{
        // 1st pass: transpose and convolve the columns
        conv_y_t.compute_root();
        conv_y_t.split(y, yo, yi, 8).parallel(yo).vectorize(x, 8);

        // 2nd pass: transpose and convolve the rows.
        conv_y.compute_root();
        conv_x.split(y, yo, yi, 8).parallel(yo).vectorize(x, 8);
      //}
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(SeparableConvolution2d, SeparableConvolution2d,
                          DO::Shakti::HalideBackend::SeparableConvolution2d)
