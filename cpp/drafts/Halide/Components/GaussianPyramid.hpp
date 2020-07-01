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
#include <drafts/Halide/Components/GaussianKernel.hpp>
#include <drafts/Halide/Components/SeparableConvolutionComponent.hpp>


namespace {

  using namespace Halide;

  template <typename Input, typename Output>
  auto gaussian_convolution(Input& input, Output& output,
                            GaussianKernelComponent& gaussian,
                            SeparableConvolutionComponent& separable_conv_2d)
  {
    separable_conv_2d.generate(
        input, gaussian.kernel, gaussian.kernel_size, gaussian.kernel_shift,
        gaussian.kernel, gaussian.kernel_size, gaussian.kernel_shift, output);
  }

  template <typename Input, typename Output>
  auto gaussian_octave(Input& input, std::vector<Output>& outputs,
                       std::vector<GaussianKernel>& gaussians,
                       SeparableConvolutionComponent& separable_conv_2d)
  {
    for (auto s = 0u; s < gaussians.size(); ++s)
    {
      auto& g = gaussians[s];
      if (s == 0)
        separable_conv_2d.generate(input,                                    //
                                   g.kernel, g.kernel_size, g.kernel_shift,  //
                                   g.kernel, g.kernel_size, g.kernel_shift,  //
                                   outputs[s]);
      else
        separable_conv_2d.generate(outputs[s - 1],                            //
                                   g.kernel, g.kernel_size, g.kernel_shift,  //
                                   g.kernel, g.kernel_size, g.kernel_shift,  //
                                   outputs[s]);
      outputs[s].compute_root();
    }
  }

}  // namespace
