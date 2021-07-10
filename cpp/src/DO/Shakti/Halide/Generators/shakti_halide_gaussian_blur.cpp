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

  class GaussianBlur : public Halide::Generator<GaussianBlur>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Buffer<float>> input{"f", 2};
    Input<float> sigma{"sigma"};
    Input<int> truncation_factor{"truncation_factor"};

    Output<Buffer<float>> output{"conv_f", 2};

    Var x{"x"}, y{"y"}, xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};
    Var i{"i"};

    void generate()
    {
      const auto w = input.width();
      const auto h = input.height();

      auto kernel_size = cast<int>(2 * sigma * truncation_factor + 1);
      kernel_size = select(kernel_size % 2 == 0, kernel_size + 1, kernel_size);
      const auto kernel_shift = -kernel_size / 2;

      // Define the unnormalized gaussian function.
      auto gaussian_unnormalized = Func{"gaussian_unnormalized"};
      gaussian_unnormalized(i) = exp(-(i * i) / (2 * sigma * sigma));

      // Define the summation variable `k` defined on a summation domain.
      auto k = RDom(kernel_shift, kernel_size);
      // Calculate the normalization factor by summing with the summation
      // variable.
      auto normalization_factor = sum(gaussian_unnormalized(k));

      auto gaussian = Func{"gaussian"};
      gaussian(i) = gaussian_unnormalized(i) / normalization_factor;

      // 1st pass: transpose and convolve the columns.
      auto input_t = Func{"input_transposed"};
      input_t(y, x) = input(x, y);

      auto conv_y_t = Func{"conv_y_t"};
      conv_y_t(x, y) = sum(input_t(clamp(x + k, 0, h - 1), y) * gaussian(k));


      // 2nd pass: transpose and convolve the rows.
      auto conv_y = Func{"conv_y"};
      conv_y(x, y) = conv_y_t(y, x);

      auto& conv_x = output;
      conv_x(x, y) = sum(conv_y(clamp(x + k, 0, w - 1), y) * gaussian(k));

      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        // Compute the gaussian first.
        gaussian.compute_root();

        // 1st pass: transpose and convolve the columns
        conv_y_t.compute_root();
        conv_y_t.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y);

        // 2nd pass: transpose and convolve the rows.
        conv_x.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Halide::Target::HVX_v62,  //
                                             Halide::Target::HVX_v65,
                                             Halide::Target::HVX_v66,
                                             Halide::Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        // Compute the gaussian first.
        gaussian.compute_root();

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
        // Compute the gaussian first.
        gaussian.compute_root();

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

HALIDE_REGISTER_GENERATOR(GaussianBlur, shakti_halide_gaussian_blur)
