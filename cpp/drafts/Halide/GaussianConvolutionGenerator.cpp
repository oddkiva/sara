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
#include "SeparableConvolution2d.stub.h"


namespace {

  using namespace Halide;
  using DO::Shakti::HalideBackend::SeparableConvolution2d;

  class GaussianConvolution : public Halide::Generator<GaussianConvolution>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Buffer<float>> input{"input", 3};
    Input<float> sigma{"sigma"};
    Input<int32_t> truncation_factor{"truncation_factor"};

    Func gaussian{"gaussian"};

    Func conv_fn;

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

      const auto radius = cast<int>(0.5f * sigma * truncation_factor);
      const auto kernel_shift = -radius;
      const auto kernel_size = 2 * radius + 1;

      auto k = RDom(kernel_shift, kernel_size);
      auto normalization_factor = sum(gaussian_unnormalized(k));
      gaussian(x) = gaussian_unnormalized(x) / normalization_factor;

      auto input_padded = BoundaryConditions::repeat_edge(input);

      conv_fn = SeparableConvolution2d::generate(
          this, {input_padded, gaussian, kernel_size, kernel_shift});

      output(x, y, c) = conv_fn(x, y, c);
    }

    void schedule()
    {
      gaussian.compute_root();
      conv_fn.compute_root();
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(GaussianConvolution, GaussianConvolution)
