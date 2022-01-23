// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/Cuda/MultiArray/CudaArray.hpp>


namespace DO::Shakti::Cuda {

  auto
  compute_histogram_of_gradients(cudaTextureObject_t mag_fn,              //
                                 cudaTextureObject_t ori_fn,              //
                                 const float* x,                          //
                                 const float* y,                          //
                                 const float* scale,                      //
                                 const float scale_max,                   //
                                 int N = 36,                              //
                                 float gaussian_truncation_factor = 3.f,  //
                                 float scale_mult_factor = 1.5f)          //
      -> void
  {
#ifdef TODO
    const auto sigma_max = scale_mult_factor * scale_max;
    const auto patch_radius_max = ::Halide::cast<std::int32_t>(
        ::Halide::floor(sigma_max * gaussian_truncation_factor));

    const auto sigma = scale_mult_factor * scale;
    const auto patch_radius = ::Halide::cast<std::int32_t>(
        ::Halide::floor(sigma * gaussian_truncation_factor));

    const auto rounded_x = ::Halide::cast<std::int32_t>(::Halide::round(x));
    const auto rounded_y = ::Halide::cast<std::int32_t>(::Halide::round(y));

    // Define the maximum reduction domain.
    auto r = ::Halide::RDom(                           //
        -patch_radius_max, 2 * patch_radius_max + 1,   // x
        -patch_radius_max, 2 * patch_radius_max + 1);  // y
    // Define the shape of the reduction domain.
    //
    // The following is more accurate:
    // r.where(r.x * r.x + r.y * r.y < patch_radius * patch_radius);
    //
    // The following is however more efficient and introduces a negligible bias:
    r.where(::Halide::abs(r.x) < patch_radius &&  //
            ::Halide::abs(r.y) < patch_radius);

    // Collect the gradient magnitudes and orientations in the reduction domain.
    const auto mag = mag_fn_ext(rounded_x + r.x, rounded_y + r.y);
    const auto ori = ori_fn_ext(rounded_x + r.x, rounded_y + r.y);

    // Calculate the Gaussian weight of each gradient orientation based on the
    // distance to the patch center.
    const auto ori_weight = ::Halide::exp(  //
        -(r.x * r.x + r.y * r.y) /          //
        (2 * sigma * sigma));               //

    // Find the corresponding orientation bin for each gradient orientation.
    //
    // Re-express the orientation value in [0, 2π].
    constexpr auto two_pi = static_cast<float>(2 * M_PI);
    const auto ori_in_0_2_pi = ::Halide::select(ori < 0,       //
                                                ori + two_pi,  //
                                                ori);          //
    // Normalize in [0, 1].
    const auto ori_01_normalized = ori_in_0_2_pi / two_pi;

    // Discretize the interval in [0, 2π[ in 36 smaller intervals centered in:
    // {  0 deg,         10 deg,         20 deg, ...,    350 deg }
    // Each bin cover the following intervals:
    //  [-5, 5[,         [5, 15[,      [15, 25[, ..., [345, 355[
    //
    // Formalising a bit, this becomes:
    // { 0 rad, Δ rad, 2Δ rad, ..., (N - 1)Δ rad }
    //
    // Each interval length is equal to:
    // Δ = 2π / N
    //
    //
    // The intervals are:
    //  [     0  Δ - Δ/2,       0 Δ + Δ/2[,
    //  [     1  Δ - Δ/2,       1 Δ + Δ/2[,
    //  [     2  Δ - Δ/2,       2 Δ + Δ/2[
    //  ...
    //  [(N - 1) Δ - Δ/2, (N - 1) Δ + Δ/2[

    // It makes more sense to use interpolation.
    const auto ori_0N = ori_01_normalized * N;

    auto ori_0N_floor = ::Halide::floor(ori_0N);
    ori_0N_floor = ::Halide::select(ori_0N_floor == N, 0, ori_0N_floor);

    const auto ori_0N_ceil =
        ::Halide::select(ori_0N_floor == N - 1, 0, ori_0N_floor + 1);

    const auto ori_0N_frac = ::Halide::fract(ori_0N);

    // Accumulation rule using linear interpolation.
    const auto interpolation_weight = ::Halide::select(  //
        o == ori_0N_floor,                               //
        (1 - ori_0N_frac),                               //
        ::Halide::select(o == ori_0N_ceil,               //
                         ori_0N_frac,                    //
                         0));                            //

    return ::Halide::sum(ori_weight * mag * interpolation_weight);
#endif
  }

}  // namespace DO::Shakti::Cuda
