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

  //! @brief The number of samples to perform for each local HoG that composes
  //! the SIFT descriptor.
  static constexpr auto sample_count_per_axis_per_subpatch =
      2 * bin_length_in_scale_unit + 1;
  static constexpr auto sample_count_per_subpatch =
      sample_count_per_axis_per_subpatch * sample_count_per_axis_per_subpatch;

  //! @brief Radius of the whole image patch.
  __host__ __device__ static inline auto
  patch_radius_bound_in_pixels(float scale)
  {
    return patch_radius_bound_in_scale_unit * scale;
  }


  __constant__ float scale;

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
    auto norm = sqrtf(calculate_sum_by_reduction(s_work));

    // Normalize to make it robust to linear contrast changes.
    s_in[ty][tx][tz] /= norm;

    // Clamp histogram bin values to 0.2 to make it robust to nonlinear
    // lighting change.
    s_in[ty][tx][tz] = min(s_in[ty][tx][tz], max_bin_value);

    // Re-normalize again.
    s_work[ty][tx][tz] = s_in[ty][tx][tz] * s_in[ty][tx][tz];
    __syncthreads();
    norm = sqrtf(calculate_sum_by_reduction(s_work));
    s_in[ty][tx][tz] /= norm;

    // The SIFT descriptor is now a unit vector.
    sifts[global_index] = s_in[ty][tx][tz];
  }

  __global__ void compute_dense_upright_sift_descriptor(float* sifts,
                                                        textureObject_t gradient_mag_texture,
                                                        textureObject_t gradient_ori_texture,
                                                        int w, int h, int pitch)
  {
    static constexpr auto pi = CUDART_PI_F;
    static constexpr auto sqrt_two = CUDART_SQRT_TWO_F;
    const auto& N = subpatch_count_per_axis;
    static constexpr auto half_N = N * 0.5f;
    static constexpr auto half_N_squared = half_N * half_N;
    static constexpr auto two_sigma_inverted = 1 / (2 * half_N * half_N);
    static constexpr auto N_squared = N * N;

    const auto p = coords<3>();
    if (p.x() >= w || p.y() >= h || p.z() >= N_squared)
      return;
    const auto x = static_cast<float>(p.x());
    const auto y = static_cast<float>(p.y());


    // SIFT is 3D descriptors encoding:
    // - local HoG for each of the NxN overlapping subpatches,
    // - each HoG discretizing the angles in O = 8 bins.
    //
    // Which subpatch (i, j) are we in?
    const auto i = p.z() / N * O;
    const auto j = p.z() - i * N;

    // The center of the subpatch i, j in image coordinates.
    const auto x = x + (j - half_N + 0.5f) * bin_length_in_scale_unit * scale;
    const auto y = y + (i - half_N + 0.5f) * bin_length_in_scale_unit * scale;

    
    // Calculate the local histogram of gradients h[i, j] that composes the SIFT
    // descriptor.
    auto h = Vector<float, 8>::Zero();

#pragma unroll
    for (auto r = 0; r < sample_count_per_subpatch; ++r)
    {
      // Retrieve the normalized integral coordinates (un, vn).
      const auto vn =
          r / sample_count_per_axis_per_subpatch - bin_length_in_scale_unit;
      const auto un = r - ry * sample_count_per_axis_per_subpatch -
                      bin_length_in_scale_unit;
      static_assert(std::is_same_v<decltype(un), const float>);
      static_assert(std::is_same_v<decltype(vn), const float>);

      const auto square_normalized_radius = un * un + vn * vn;
      const auto spatial_weight =
          expf(-square_normalized_radius * two_gradient_sigma_inverted);

      const auto u = un * bin_length_in_scale_unit * scale;
      const auto v = vn * bin_length_in_scale_unit * scale;

      const auto mag = tex2D<float>(gradient_mag_texture, x + u , y + v);
      const auto ori = tex2D<float>(gradient_ori_texture, x + u , y + v);

      ori = ori < 0.f ? ori + 2.f * pi : ori;
      ori *= float(O) / (2.f * pi);

      const auto dori = __saturatef(fabs(ori - floorf(ori)));
      const auto dx = fabs(vn) / bin_length_in_scale_unit;
      const auto dy = fabs(un) / bin_length_in_scale_unit;
      const auto wx = 1 - dx;
      const auto wy = 1 - dy;
      const auto wo = 1 - dori;

      const auto o1 = int(ori);
      const auto o2 = o == 7 ? 0 : o1 + 1;
      h[o1] += (1 - dx) * (1 - dy) * (1 - do) * spatial_weight * mag;
      h[o2] += dx * dy * dori * spatial_weight * mag;
    }

    // Copy the histogram
    //
    // Check the pitch and so on but the idea seems to be here.
    const auto global_index = (i * N * O + j) * h * pitch + p.y() * pitch + x;
    for (auto i = 0; i < 8; ++i)
      sifts[global_index + i] = h[i];
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
