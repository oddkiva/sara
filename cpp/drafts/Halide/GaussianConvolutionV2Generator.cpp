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

#include "MyHalide.hpp"
#include "SeparableConvolutionComponent.hpp"


namespace {

  using namespace Halide;

  class GaussianConvolutionV2 : public Halide::Generator<GaussianConvolutionV2>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Buffer<float>> input{"input", 3};
    Input<float> sigma{"sigma"};
    Input<int32_t> truncation_factor{"truncation_factor"};

    SeparableConvolutionComponent separable_conv_2d;
    Func gaussian{"gaussian"};

    Output<Buffer<float>> output{"input_convolved", 3};

    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"}, c{"c"};
    Var xo{"xo"}, yo{"yo"};
    Var xi{"xi"}, yi{"yi"};
    //! @}

    void generate()
    {
      // Define the unnormalized gaussian function.
      auto gaussian_unnormalized = Func{"gaussian_unnormalized"};
      gaussian_unnormalized(x) = exp(-(x * x) / (2 * sigma * sigma));

      auto kernel_size = cast<int>(2 * sigma * truncation_factor + 1);
      kernel_size = select(kernel_size % 2 == 0, kernel_size + 1, kernel_size);
      const auto kernel_shift = -kernel_size / 2;

      auto k = RDom(kernel_shift, kernel_size);
      auto normalization_factor = sum(gaussian_unnormalized(k));
      gaussian(x) = gaussian_unnormalized(x) / normalization_factor;

      separable_conv_2d.generate(input, gaussian, kernel_size, kernel_shift, output);
    }

    void schedule()
    {
      gaussian.compute_root();
      separable_conv_2d.schedule(get_target(), tile_x, tile_y, output);
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(GaussianConvolutionV2, shakti_gaussian_convolution_v2)
