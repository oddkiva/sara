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

#include <DO/Shakti/Halide/Components/SeparableConvolution.hpp>


namespace {

  using namespace Halide;

  class SeparableConvolution2d : public Generator<SeparableConvolution2d>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Buffer<float>> input{"input", 4};
    Input<Buffer<float>> kernel{"kernel", 1};
    Input<int32_t> kernel_size{"kernel_size"};
    Input<int32_t> kernel_shift{"kernel_shift"};

    Output<Buffer<float>> output{"input_convolved", 4};

    DO::Shakti::HalideBackend::SeparableConvolution separable_conv_2d;

    void generate()
    {
      const auto w = input.dim(0).extent();
      const auto h = input.dim(1).extent();

      separable_conv_2d.generate(
          input,
          kernel, kernel_size, kernel_shift,
          kernel, kernel_size, kernel_shift,
          output, w, h);
    }

    void schedule() 
    {
      separable_conv_2d.schedule(get_target(), tile_x, tile_y, output);
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(SeparableConvolution2d,
                          shakti_separable_convolution_2d_cpu,
                          DO::Shakti::HalideBackend::SeparableConvolution2d)
HALIDE_REGISTER_GENERATOR(SeparableConvolution2d,
                          shakti_separable_convolution_2d_gpu,
                          DO::Shakti::HalideBackend::SeparableConvolution2d)
