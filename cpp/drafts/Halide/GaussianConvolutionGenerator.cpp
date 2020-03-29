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

#include "SeparableConvolution2d.stub.h"


namespace {

  using namespace Halide;
  using DO::Shakti::HalideBackend::SeparableConvolution2d;

  class GaussianConvolution : public Halide::Generator<GaussianConvolution>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Func> input{"input", Float(32), 3};
    Input<float> sigma{"sigma"};
    Input<int32_t> truncation_factor{"truncation_factor"};

    Func gaussian{"gaussian"};

    Output<Func> output{"input_convolved", Float(32), 3};

    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"};
    Var xo{"xo"}, yo{"yo"};
    Var xi{"xi"}, yi{"yi"};
    //! @}

    void generate()
    {
      // Define the unnormalized gaussian function.
      auto gaussian_unnormalized = Func{"gaussian_unnormalized"};
      gaussian_unnormalized(x) = exp(-(x * x) / (2 * sigma * sigma));

      const auto radius = cast<int>(sigma / 2) * truncation_factor;
      const auto kernel_shift = -radius;
      const auto kernel_size = 2 * radius + 1;

      auto k = RDom(kernel_shift, kernel_size);
      auto normalization_factor = sum(gaussian_unnormalized(k));
      gaussian(x) = gaussian_unnormalized(x) / normalization_factor;

      output = SeparableConvolution2d::generate(
          this, {input, gaussian, kernel_shift, kernel_size});
    }

    void schedule()
    {
      gaussian.compute_root();
      //conv.schedule();
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(GaussianConvolution, GaussianConvolution)
