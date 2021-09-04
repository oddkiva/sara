// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#include <DO/Shakti/Halide/MyHalide.hpp>


namespace DO::Shakti::Halide {

  struct Operation
  {
    virtual auto run() -> void = 0;
  };

  //! @brief Subtraction.
  struct SubtractOp : Operation
  {
    //! @brief Inputs.
    //! @{
    ::Halide::Runtime::Buffer<float> a;
    ::Halide::Runtime::Buffer<float> b;
    //! @}

    //! @brief Ouput
    //! output = a - b
    ::Halide::Runtime::Buffer<float> output;
    bool gpu;

    auto run() -> void override;
  };

  struct ScaleOp : Operation
  {
    ::Halide::Runtime::Buffer<float> input;
    ::Halide::Runtime::Buffer<float> output;
    bool gpu;

    auto run() -> void override;
  };

  struct ReduceOp : Operation
  {
    ::Halide::Runtime::Buffer<float> input;
    ::Halide::Runtime::Buffer<float> output;
    bool gpu;

    auto run() -> void override;
  };

  struct ReduceOp : Operation
  {
    ::Halide::Runtime::Buffer<float> input;
    ::Halide::Runtime::Buffer<float> output;
    bool gpu;

    auto run() -> void override;
  };

  struct EnlargeOp : Operation
  {
    ::Halide::Runtime::Buffer<float> input;
    ::Halide::Runtime::Buffer<float> output;
    bool gpu;
    auto run() -> void override;
  };

  struct EnlargeOp : Operation
  {
    ::Halide::Runtime::Buffer<float> input;
    ::Halide::Runtime::Buffer<float> output;
    bool gpu;
    auto run() -> void override;
  };
  //! @}

  //! @brief Batched convolution function.
  struct ConvolveOp : Operation
  {
    ::Halide::Runtime::Buffer<float> src;
    ::Halide::Runtime::Buffer<float> kernel;
    ::Halide::Runtime::Buffer<float> dst;
    bool gpu;
    auto run() -> void override;
  };

  //! @brief Gaussian convolution.
  struct ConvolveOp : Operation
  {
    ::Halide::Runtime::Buffer<float> src;
    ::Halide::Runtime::Buffer<float>& dst;
    float sigma;
    int truncation_factor;
    bool gpu;
    auto run() -> void override;
  };

  struct PolarGradient : Operation
  {
    ::Halide::Runtime::Buffer<float> in;
    ::Halide::Runtime::Buffer<float> mag;
    ::Halide::Runtime::Buffer<float> ori;
    bool gpu;
    auto run() -> void override;
  };

  struct DominantGradientOrientations : Operation
  {
    ::Halide::Runtime::Buffer<float> gradient_magnitudes;
    ::Halide::Runtime::Buffer<float> gradient_orientations;
    ::Halide::Runtime::Buffer<float> x;
    ::Halide::Runtime::Buffer<float> y;
    ::Halide::Runtime::Buffer<float> scale;
    float scale_upper_bound;
    ::Halide::Runtime::Buffer<bool> peak_map;
    ::Halide::Runtime::Buffer<float> peak_residuals;

    int num_orientation_bins = 36;
    float gaussian_truncation_factor = 3.f;
    float scale_multiplying_factor = 1.5f;
    float peak_ratio_thres = 0.8f;
    bool gpu;
    auto run() -> void override;
  };

}  // namespace DO::Shakti::Halide
