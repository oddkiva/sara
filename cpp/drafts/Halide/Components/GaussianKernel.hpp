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


namespace DO::Shakti::HalideBackend {

  using namespace Halide;

  struct GaussianKernel
  {
    //! @brief Variables.
    Var x{"x"};

    Func kernel{"gaussian"};
    Expr kernel_size;
    Expr kernel_shift;

    void generate(Expr sigma, Expr truncation_factor)
    {
      // Define the unnormalized gaussian function.
      auto gaussian_unnormalized = Func{"gaussian_unnormalized"};
      gaussian_unnormalized(x) = exp(-(x * x) / (2 * sigma * sigma));

      kernel_size = cast<int>(2 * sigma * truncation_factor + 1);
      kernel_size = select(kernel_size % 2 == 0, kernel_size + 1, kernel_size);

      kernel_shift = -kernel_size / 2;

      auto k = RDom(kernel_shift, kernel_size);
      auto normalization_factor = sum(gaussian_unnormalized(k));

      kernel(x) = gaussian_unnormalized(x) / normalization_factor;
    }

    void schedule()
    {
      kernel.compute_root();
    }
  };

}  // namespace DO::Shakti::HalideBackend
