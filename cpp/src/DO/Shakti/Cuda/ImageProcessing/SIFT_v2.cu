// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <math_constants.h>

#include <DO/Shakti/Cuda/ImageProcessing.hpp>
#include <DO/Shakti/Cuda/ImageProcessing/Kernels/Globals.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/MultiArray/Offset.hpp>

#include <DO/Shakti/Cuda/Utilities/Timer.hpp>


namespace DO::Shakti {

  __constant__ float bin_scale_unit_length;
  __constant__ float max_bin_value;
  __constant__ float sigma;
  __constant__ float sample_count_per_axis_per_subpatch;

  //! @brief Robustify descriptor w.r.t. illumination changes.
  __global__ void normalize(float* sifts, int w, int h, int pitch)
  {
    static constexpr auto dim = 128;
    const auto p = coords<3>();
    if (p.x() >= w || p.y() >= h || p.z() >= dim)
      return;

    const auto global_index = p.z() * h * pitch + p.y() * pitch + x;
    const auto& tx = threadIdx.x;
    const auto& ty = threadIdx.y;
    const auto& tz = threadIdx.z;

    __shared__ float s_in[4][2][128];
    __shared__ float s_norm[4][2][128];

    const auto value_unnormalized = sifts[global_index];

    // Accumulate the unnormalized bin values in shared memory.
    s_in[ty][tx][tz] = value_unnormalized;
    // Accumulate the square values.
    s_work[ty][tx][tz] = value_unnormalized * value_unnormalized;
    __syncthreads();

    // Calculate the Euclidean norm of the descriptors as follows.
    for (auto i  = 64; i > 1; i >>= 1)
      if (tz < i)
        s_work[ty][tx][tz] += s_work[ty][tx][tz + i];
    __syncthreads();
    const auto norm = __sqrtf(s_work[ty][tx][0]);

    // Normalize to make it robust to linear contrast changes.
    s_in[ty][tx][tz] /= norm;

    // Clamp histogram bin values to 0.2 to make it robust to nonlinear
    // lighting change.
    s_in[ty][tx][tz] = min(s_in[ty][tx][tz], max_bin_value);

    // Re-normalize again.
    s_work[ty][tx][tz] = s_in[ty][tx][tz] * s_in[ty][tx][tz];
    __syncthreads();
    for (auto i  = 64; i > 1; i >>= 1)
      if (tz < i)
        s_work[ty][tx][tz] += s_work[ty][tx][tz + 64];
    __syncthreads();
    const auto norm = __sqrtf(s_work[ty][tx][0]);
    s_in[ty][tx][tz] /= norm;

    sifts[global_index] = s_in[ty][tx][tz];
  }

  __global__ void compute_dense_upright_sift_descriptor(float* sifts, int w,
                                                        int h, int pitch)
  {
    static constexpr auto dim = N * N * O;
    static constexpr auto pi = static_cast<float>(CUDART_PI);
    static constexpr auto sqrt_two = static_cast<float>(CUDART_SQRT_TWO);
    static constexpr auto patch_radius = (N + 1) * 0.5f;

    const auto p = coords<3>();
    if (p.x() >= w || p.y() >= h || p.z() >= dim)
      return;

    const auto i = p.z() / N * O;
    const auto j = (p.z() - i * N * O) / O;
    const auto o = p.z() - i * N * O - j * O;

    const auto l = bin_scale_unit_length * sigma;

    const auto r = sqrt_two * bin_scale_unit_length * patch_radius;
    const auto rounded_r = static_cast<int>(r);
    const auto total_samples =
        sample_count_per_axis_per_subpatch * sample_count_per_axis_per_subpatch;

    // Calculate a histogram in each pixel.
    //
    // TODO: sample a fixed number of points instead of scanning every pixel of
    // the patch.

    auto bin_value = 0.f;
#pragma unroll
    for (auto uv = 0; uv < total_samples; ++uv)
    {
      const auto v = uv / sample_count_per_axis_per_subpatch;
      const auto u = uv - v * sample_count_per_axis_per_subpatch;
      // TODO: locate the coordinates
      // Refresh ourselves about the patch geometry.

      const auto weight = exp(-(u * u + v * v) / (2.f * powf(N / 2.f, 2)));
      auto grad = tex2D(in_float2_texture, x + u, y + v);
      auto mag = grad.x;
      auto ori = grad.y;

      ori = ori < 0.f ? ori + 2.f * pi : ori;
      ori *= float(O) / (2.f * pi);
    }

    const auto global_index = p.z() * h * pitch + p.y() * pitch + x;
    sifts[global_index] = bin_value;
  }

  template <int N, int O>
  auto
  compute_dense_upright_sift_descriptor(const TextureArray<Vector2f>& gradients)
      -> MultiArray<float, 3>
  {
    static constexpr auto dim = N * N * O;
    const auto h = gradients.sizes().y();
    const auto w = gradients.sizes().x();
    auto sifts = MultiArray<float, 3>{{w, h, dim}};

    const auto block_size = dim3(16, 16, 4);
    const auto grid_size =
        grid_size_2d({w, h}, sifts.padded_width(), block_size);
    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float2_texture, gradients));
    compute_dense_upright_sift_descriptor<N, O><<<grid_size, block_size>>>(
        sifts.data(), sifts.width(), sifts.height(), sifts.padded_width());
    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float2_texture));
    return sifts;
  }

}  // namespace DO::Shakti
