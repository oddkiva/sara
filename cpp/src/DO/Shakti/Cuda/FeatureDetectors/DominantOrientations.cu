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

#include <DO/Shakti/Cuda/FeatureDetectors/Octave.hpp>
#include <DO/Shakti/Cuda/MultiArray/CudaArray.hpp>

#include <math_constants.h>


namespace DO::Shakti::Cuda {

  static constexpr auto scale_mult_factor = 1.5f;
  static constexpr auto gaussian_truncation_factor = 3;
  static constexpr auto orientation_bin_count = 36;
  static constexpr auto sample_count_per_dim = 2 * gaussian_truncation_factor;
  static constexpr auto sample_count = sample_count_per_dim *  //
                                       sample_count_per_dim;
  static constexpr auto start_sample_coord =
      -scale_mult_factor * gaussian_truncation_factor;

  // So we sample 36 gradients. That should be plenty.
  static_assert(sample_count_per_dim == 2 * 3);
  static_assert(sample_count == 36);

  // Not a power of two. Does not matter.
  static constexpr auto tile_o = 36;
  static constexpr auto tile_i = 1024 / tile_o;

  __global__ auto compute_histogram_of_gradients(cudaTextureObject_t mag_fn,  //
                                                 cudaTextureObject_t ori_fn,  //
                                                 float* histogram,
                                                 const float* x_arr,  //
                                                 const float* y_arr,  //
                                                 const float* s_arr,
                                                 int extremum_count,
                                                 int orientation_pitch) -> void
  {
    const auto o = blockIdx.x * blockDim.x + threadIdx.x;
    const auto i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= extremum_count || o >= orientation_bin_count)
      return;

    const auto x = x_arr[i];
    const auto y = y_arr[i];
    const auto sigma = s_arr[i];

    const auto& to = threadIdx.x;
    const auto& ti = threadIdx.y;
    const auto t = ti * tile_o + to;

    __shared__ float s_hist[tile_o * tile_i];

    // Initialize the histogram of gradients to zero.
    s_hist[t] = 0;
    __syncthreads();

    static constexpr auto x0 = start_sample_coord;
    static constexpr auto y0 = start_sample_coord;
    const auto two_sigma_square_inv = fdividef(1.f, 2.f * (sigma * sigma));

    const auto two_pi = 2 * CUDART_PI;
    const auto ori_norm_factor = orientation_bin_count / two_pi;

#pragma unroll
    for (auto dy = 0; dy < sample_count_per_dim; ++dy)
    {
      const auto y_rel_in_scale_units = y0 + dy;
      const auto y_abs_in_pixels = y + y_rel_in_scale_units * sigma;
      for (auto dx = 0; dx < sample_count_per_dim; ++dx)
      {
        const auto x_rel_in_scale_units = x0 + dx;
        const auto x_abs_in_pixels = x + x_rel_in_scale_units * sigma;

        const auto square_dist =
            x_rel_in_scale_units * x_rel_in_scale_units +  //
            y_rel_in_scale_units * y_rel_in_scale_units;

        // Get the gradient.
        const auto gradient_norm =
            tex2D<float>(mag_fn, x_abs_in_pixels, y_abs_in_pixels);

        // Get the spatial weight in relative scale units.
        const auto w_spatial = expf(-square_dist);

        // Calculate the total weight increment.
        [[maybe_unused]] const auto w_increment = w_spatial * gradient_norm;

        // Now find out the two closest orientation bins which we will
        // distribute the weight increment to.
        //
        // 1. Get the absolute orientation, which is in [-Pi, Pi].
        const auto ori = tex2D<float>(ori_fn, x_abs_in_pixels, y_abs_in_pixels);
        // 2. Retransform it to [0, 2*Pi].
        [[maybe_unused]] auto ori_normalized = ori < 0 ? ori + two_pi : ori;
        // 3. Renormalize in [0, N].
        ori_normalized = ori * ori_norm_factor;

        // 4.a) Get the fractional and integral part.
        float ori_int;
        const auto ori_frac = modff(ori, &ori_int);
        // 4.b) Important: clamp the orientation bin.
        ori_int = fminf(ori_int, orientation_bin_count);

        // 5. Now we can calculate the two closest orientation bins.
        const auto ori_bin_0 = int(ori_int);
        const auto ori_bin_1 =
            ori_bin_0 == orientation_bin_count ? 0 : ori_bin_0 + 1;

        // 6. Distribute the total weight increment by linear interplation to
        //    the two closest orientation bins.
        const auto w0 = w_spatial * (1 - ori_frac);
        const auto w1 = w_spatial * ori_frac;

        const auto t0 = ti * tile_o + ori_bin_0;
        const auto t1 = ti * tile_o + ori_bin_1;

        s_hist[t0] = atomicAdd(&s_hist[t0], w0);
        s_hist[t1] = atomicAdd(&s_hist[t1], w1);
      }
    }

    const auto gi = i * orientation_pitch + o;
    histogram[gi] = s_hist[ti * tile_o + to];
  }

  __global__ auto
  compute_histogram_of_gradients(cudaTextureObject_t gaussian_fn,  //
                                 float* histogram,
                                 const float* x_arr,  //
                                 const float* y_arr,  //
                                 const int* s_arr, const int* s_layer_arr,
                                 int extremum_count, int orientation_pitch)
      -> void
  {
    const auto o = blockIdx.x * blockDim.x + threadIdx.x;
    const auto i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= extremum_count || o >= orientation_bin_count)
      return;

    const auto x = x_arr[i];
    const auto y = y_arr[i];
    [[maybe_unused]] const auto s = s_layer_arr[i];
    const auto sigma = s_arr[i];

    const auto& to = threadIdx.x;
    const auto& ti = threadIdx.y;
    const auto t = ti * tile_o + to;

    __shared__ float s_hist[tile_o * tile_i];

    // Initialize the histogram of gradients to zero.
    s_hist[t] = 0;
    __syncthreads();

    static constexpr auto x0 = start_sample_coord;
    static constexpr auto y0 = start_sample_coord;
    const auto two_sigma_square_inv = fdividef(1.f, 2.f * (sigma * sigma));

    const auto two_pi = 2 * CUDART_PI;
    const auto ori_norm_factor = orientation_bin_count / two_pi;

    auto gradient = [gaussian_fn](float x, float y, int s) {
      const auto Ix1 = tex2DLayered<float>(gaussian_fn, x - 1, y, s);
      const auto Ix2 = tex2DLayered<float>(gaussian_fn, x + 1, y, s);

      const auto Iy1 = tex2DLayered<float>(gaussian_fn, x - 1, y, s);
      const auto Iy2 = tex2DLayered<float>(gaussian_fn, x + 1, y, s);

      const auto g = Vector2f{0.5f * (Ix2 - Ix1), 0.5f * (Iy2 - Iy1)};

      const auto mag_xys = sqrtf(g.x() * g.x() + g.y() * g.y());
      const auto ori_xys = atan2f(g.y(), g.x());

      return std::make_pair(mag_xys, ori_xys);
    };

#pragma unroll
    for (auto dy = 0; dy < sample_count_per_dim; ++dy)
    {
      const auto y_rel_in_scale_units = y0 + dy;
      const auto y_abs_in_pixels = y + y_rel_in_scale_units * sigma;
      for (auto dx = 0; dx < sample_count_per_dim; ++dx)
      {
        const auto x_rel_in_scale_units = x0 + dx;
        const auto x_abs_in_pixels = x + x_rel_in_scale_units * sigma;

        const auto square_dist =
            x_rel_in_scale_units * x_rel_in_scale_units +  //
            y_rel_in_scale_units * y_rel_in_scale_units;

        // Get the gradient.
        auto [gradient_norm, ori] =
            gradient(gaussian_fn, x_abs_in_pixels, y_abs_in_pixels);

        // Get the spatial weight in relative scale units.
        const auto w_spatial = expf(-square_dist);

        // Calculate the weight increment.
        [[maybe_unused]] const auto w_increment = w_spatial * gradient_norm;

        // Now find out the two closest orientation bins. We will distribute the
        // weight increment to both bins.
        //
        // 1. Get the absolute orientation, which is in [-Pi, Pi].
        // 2. Retransform it to [0, 2*Pi].
        [[maybe_unused]] auto ori_normalized = ori < 0 ? ori + two_pi : ori;
        // 3. Renormalize in [0, N].
        ori_normalized = ori * ori_norm_factor;

        // 4.a) Get the fractional and integral part.
        float ori_int;
        const auto ori_frac = modff(ori, &ori_int);
        // 4.b) Important: clamp the orientation bin.
        ori_int = fminf(ori_int, orientation_bin_count);

        // 5. Now we can calculate the two closest orientation bins.
        const auto ori_bin_0 = int(ori_int);
        const auto ori_bin_1 =
            ori_bin_0 == orientation_bin_count ? 0 : ori_bin_0 + 1;

        // 6. Distribute the total weight increment by linear interplation to
        //    the two closest orientation bins.
        const auto w0 = w_spatial * (1 - ori_frac);
        const auto w1 = w_spatial * ori_frac;

        const auto t0 = ti * tile_o + ori_bin_0;
        const auto t1 = ti * tile_o + ori_bin_1;

        s_hist[t0] = atomicAdd(&s_hist[t0], w0);
        s_hist[t1] = atomicAdd(&s_hist[t1], w1);
      }
    }

    const auto gi = i * orientation_pitch + o;
    histogram[gi] = s_hist[ti * tile_o + to];
  }

  auto compute_histogram_of_gradients(const Octave<float>& gaussians,  //
                                      const float* x,                  //
                                      const float* y,                  //
                                      const float* s) -> void
  {
    if (gaussians.texture_object() == 0)
      throw std::runtime_error{"Error: texture object must be initialized!"};
  }

}  // namespace DO::Shakti::Cuda
