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
#include <drafts/Halide/Components/GaussianKernelComponent.hpp>
#include <drafts/Halide/Components/SeparableConvolutionComponent.hpp>


namespace {

  using namespace Halide;

  class GaussianConvolutionV2 : public Halide::Generator<GaussianConvolutionV2>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Buffer<float>> input{"input", 4};
    Input<float> sigma{"sigma"};
    Input<int32_t> truncation_factor{"truncation_factor"};

    GaussianKernelComponent gaussian;
    SeparableConvolutionComponent separable_conv_2d;

    Output<Buffer<float>> output{"input_convolved", 4};

    void generate()
    {
      gaussian.generate(sigma, truncation_factor);

      const auto w = input.dim(0).extent();
      const auto h = input.dim(1).extent();

      separable_conv_2d.generate(
          input,
          gaussian.kernel, gaussian.kernel_size, gaussian.kernel_shift,
          gaussian.kernel, gaussian.kernel_size, gaussian.kernel_shift,
          output, w, h);
    }

    void schedule()
    {
      gaussian.schedule();
      separable_conv_2d.schedule(get_target(), tile_x, tile_y, output);
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(GaussianConvolutionV2, shakti_gaussian_convolution_v2)
