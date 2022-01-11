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

  //! @brief SIFT build histograms from a grid of NxN patches of image
  //! gradients.
  static constexpr auto subpatch_count_per_axis = 4;
  //! @brief Each histogram of gradients consists of O = 8 orientation bins.
  /*!
   *  Each bin are centered around the orientations:
   *  bin index   = [0,   1,    2,    3, 4,    5,    6,    7]
   *  orientation = [0, π/4,  π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
   */
  static constexpr auto orientation_bin_count = 8;
  //! @brief the dimension of the SIFT descriptor.
  static constexpr auto SIFT_dimension =
      subpatch_count_per_axis * subpatch_count_per_axis * orientation_bin_count;
  static_assert(SIFT_dimension == 128);

  //! @brief Each image subpatch has a circular radius equal to:
  static constexpr auto bin_length_in_scale_unit = 3.f;
  //! @brief Each image subpatch has a circular diameter equal to:
  static constexpr auto subpatch_diameter_in_scale_unit =
      2 * bin_length_in_scale_unit + 1;
  //! @brief Each image patch has a circular radius equal to:
  static constexpr auto patch_radius_in_scale_unit =
      bin_length_in_scale_unit * (subpatch_count_per_axis + 1) / 2;
  //! @brief Now because we sample on a square patch, we need to calculate the
  //! radius of the square, but not all pixel in the square patch will
  //! contribute to the histogram. So this a bound to properly speak.
  static constexpr auto patch_radius_bound_in_scale_unit =
      patch_radius_in_scale_unit * static_cast<float>(CUDART_SQRT_TWO);

  //! @brief The magic constant value to enforce invariance to nonlinear
  //! illumination changes as referenced in the paper.
  static constexpr auto max_bin_value = 0.2f;

  //! @brief Radius of the whole image patch.
  __host__ __device__ static inline auto
  patch_radius_bound_in_pixels(float scale)
  {
    return patch_radius_bound_in_scale_unit * scale;
  }


  __constant__ float sigma;

  //! @brief Robustify descriptor w.r.t. illumination changes.
  __global__ void normalize(float* sifts, int w, int h, int pitch)
  {
    const auto p = coords<3>();
    if (p.x() >= w || p.y() >= h || p.z() >= SIFT_dimension)
      return;

    const auto global_index = p.z() * h * pitch + p.y() * pitch + x;
    const auto& tx = threadIdx.x;
    const auto& ty = threadIdx.y;
    const auto& tz = threadIdx.z;

    __shared__ float s_in[2][4][128];
    __shared__ float s_work[2][4][128];

    const auto value_unnormalized = sifts[global_index];

    // Accumulate the unnormalized bin values in shared memory.
    s_in[ty][tx][tz] = value_unnormalized;
    // Accumulate the square values.
    s_work[ty][tx][tz] = value_unnormalized * value_unnormalized;
    __syncthreads();

    auto calculate_sum_by_reduction =
        [](volatile float s_work[2][4][SIFT_dimension]) {
          for (auto i = 64; i > 0; i >>= 1)
          {
            if (tz < i)
              s_work[ty][tx][tz] += s_work[ty][tx][tz + i];
            __syncthreads();
          }
          const float sum = s_work[ty][tx][0];
          return sum;
        };

    // Calculate the Euclidean norm of the descriptors as follows.
    auto norm = __sqrtf(calculate_sum_by_reduction(s_work));

    // Normalize to make it robust to linear contrast changes.
    s_in[ty][tx][tz] /= norm;

    // Clamp histogram bin values to 0.2 to make it robust to nonlinear
    // lighting change.
    s_in[ty][tx][tz] = min(s_in[ty][tx][tz], max_bin_value);

    // Re-normalize again.
    s_work[ty][tx][tz] = s_in[ty][tx][tz] * s_in[ty][tx][tz];
    __syncthreads();
    norm = __sqrtf(calculate_sum_by_reduction(s_work));
    s_in[ty][tx][tz] /= norm;

    // The SIFT descriptor is now a unit vector.
    sifts[global_index] = s_in[ty][tx][tz];
  }

  __global__ void compute_dense_upright_sift_descriptor(float* sifts, int w,
                                                        int h, int pitch)
  {
    static constexpr auto pi = static_cast<float>(CUDART_PI);
    static constexpr auto sqrt_two = static_cast<float>(CUDART_SQRT_TWO);
    static constexpr auto half_N = subpatch_count_per_axis * 0.5f;
    static constexpr auto half_N_squared = half_N * half_N;
    static constexpr auto two_gradient_sigma_inverted =
        1 / (2 * half_N * half_N);

    const auto p = coords<3>();
    if (p.x() >= w || p.y() >= h || p.z() >= SIFT_dimension)
      return;

    // SIFT is 3D descriptors encoding:
    // - local HoG for each of the NxN overlapping subpatches,
    // - each HoG discretizing the angles in O = 8 bins.
    //
    // Which subpatch (i, j) are we in?
    const auto& N = subpatch_count_per_axis;
    const auto& O = orientation_bin_count;
    const auto i = p.z() / N * O;
    const auto j = (p.z() - i * N * O) / O;
    // Which orientation bin o are we in?
    const auto o = p.z() - i * N * O - j * O;

    const auto subpatch_radius = bin_scale_unit_length * sigma;

    const auto patch_radius_bound =
        sqrt_two * bin_scale_unit_length * patch_radius;
    const auto rounded_r = static_cast<int>(patch_radius_bound);
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

      const auto r2 = static_cast<float>(u * u + v * v);

      const auto weight = expf(-r2 * two_gradient_sigma_inverted);
      auto grad = tex2D(in_float2_texture, key_x + u, key_y + v);
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
