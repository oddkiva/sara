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

// Convolving in batch the input image does not seem very fast.
// Rather convolving sequentially seems much faster if we base ourselves from
// the computation time spent in the Halide implementation.


#define BOOST_TEST_MODULE "Shakti/CUDA/FeatureDetectors/Gaussian Convolution"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Math/UsualFunctions.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/Octave.hpp>
#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/Utilities/Timer.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace sc = DO::Shakti::Cuda;


namespace Shakti::Cuda::Gaussian {

  static constexpr auto kernel_max_radius = 24;
  static constexpr auto sdata_rows = 3 * kernel_max_radius;
  static constexpr auto tile_size = 1024 / kernel_max_radius;

  static constexpr auto kernel_count_max = 16;
  static constexpr auto kernel_capacity = kernel_max_radius * kernel_count_max;
  static_assert(kernel_capacity == 384);

  __constant__ float kernels[kernel_capacity];
  __constant__ int kernel_radii[kernel_count_max];
  __constant__ int kernel_count;
  __constant__ int kernel_radius_max;

  __global__ auto convx(const float* in, float* out, int w, int h, int pitch,
                        int kernel_index) -> void
  {
    __shared__ float sdata[tile_size * sdata_rows];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= h)
      return;

    const auto& r = kernel_radii[kernel_index];
    const auto gi = y * pitch + x;
    const auto ti = threadIdx.y * sdata_rows + threadIdx.x;

    // 1. Accumulate the shared data.
    sdata[ti] = x - r >= 0 ? in[gi - r] : in[y * pitch];
    sdata[ti + 2 * r] = x + r < w ? in[gi + r] : in[y * pitch + w - 1];
    __syncthreads();


    // Pick the right kernel.
    const auto kernel = kernels + kernel_index * kernel_radius_max;

    // 2. Convolve.
    auto val = kernel[0] * sdata[ti + r];
#pragma unroll
    for (auto k = 1; k <= r; ++k)
      val += kernel[k] * (sdata[ti + r - k] + sdata[ti + r + k]);

    out[gi] = val;
  }

  __global__ auto convy(const float* in, float* out, int w, int h, int pitch,
                        int kernel_index) -> void
  {
    __shared__ float sdata[tile_size * sdata_rows];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
      return;

    const int& r = kernel_radii[kernel_index];
    const int gi = y * pitch + x;

    // 1. Accumulate the shared data.
    const auto& tx = threadIdx.x;
    const auto& ty = threadIdx.y;
    const auto t = tx * sdata_rows + ty;
    sdata[t + r] = y - r >= 0 ? in[gi - r * pitch] : in[gi];
    sdata[t + 2 * r] = y + r < h ? in[gi + r * pitch] : in[gi];
    __syncthreads();

    // Pick the right kernel.
    const auto kernel = kernels + kernel_index * kernel_radius_max;

    // 2. Convolve.
    auto val = sdata[t + r] * kernel[0];
#pragma unroll
    for (auto k = 1; k <= r; ++k)
      val += kernel[k] * (sdata[t + r - k] + sdata[t + r + k]);

    out[gi] = val;
  }

}  // namespace Shakti::Cuda::Gaussian

namespace scg = Shakti::Cuda::Gaussian;


BOOST_AUTO_TEST_CASE(test_convolve)
{
  static constexpr auto scale_count = 3;
  static constexpr auto scale_camera = 1.f;
  static constexpr auto scale_initial = 1.6f;
  static constexpr auto gaussian_truncation_factor = 4.f;
  static const float scale_factor = std::pow(2.f, 1.f / scale_count);

  // List the scales in the discrete octave.
  auto scales = std::vector<float>(scale_count + 3);
  for (auto i = 0; i < scale_count + 3; ++i)
    scales[i] = scale_initial * std::pow(scale_factor, i);

  // Calculate the Gaussian smoothing values.
  auto sigmas = std::vector<float>(scale_count + 3);
  for (auto i = 0u; i < sigmas.size(); ++i)
  {
    sigmas[i] =
        i == 0
            ? std::sqrt(sara::square(scales[0]) - sara::square(scale_camera))
            : std::sqrt(sara::square(scales[i]) - sara::square(scales[i - 1]));
  }

  SARA_DEBUG << "scales =\n"
             << Eigen::Map<const Eigen::RowVectorXf>(scales.data(),
                                                     scales.size())
             << std::endl;
  SARA_DEBUG << "sigmas =\n"
             << Eigen::Map<const Eigen::RowVectorXf>(sigmas.data(),
                                                     sigmas.size())
             << std::endl;

  // Calculater the kernel dimensions.
  auto kernel_radii = std::vector<int>{};
  for (const auto& sigma : sigmas)
  {
    auto radius = std::round(gaussian_truncation_factor * sigma);
    if (radius == 0)
      radius = 1;
    kernel_radii.push_back(static_cast<int>(radius));
  }
  const auto kernel_radius_max = kernel_radii.back();
  SARA_DEBUG << "kernel radii =\n"
             << Eigen::Map<const Eigen::RowVectorXi>(kernel_radii.data(),
                                                     kernel_radii.size())
             << std::endl;

  // Fill the Gaussian kernels.
  auto kernels = sara::Tensor_<float, 2>{
      scale_count + 3,   //
      kernel_radius_max  //
  };
  kernels.flat_array().fill(0);

  for (auto n = 0; n < kernels.size(0); ++n)
  {
    const auto& sigma = sigmas[n];
    const auto two_sigma_squared = 2 * sara::square(sigma);
    const auto kernel_radius = kernel_radii[n];

    for (auto k = 0; k <= kernel_radius; ++k)
      kernels(n, k) = exp(-sara::square(k) / two_sigma_squared);

    auto kernel_sum = kernels(n, 0);
    for (auto k = 1; k <= kernel_radius; ++k)
      kernel_sum += 2 * kernels(n, k);

    for (auto k = 0; k <= kernel_radius; ++k)
      kernels(n, k) /= kernel_sum;
  }

  Eigen::IOFormat HeavyFmt(3, 0, ", ", ",\n", "[", "]", "[", "]");
  SARA_CHECK(Eigen::Map<const Eigen::RowVectorXf>(  //
      sigmas.data(),                                //
      sigmas.size())                                //
  );
  SARA_CHECK(kernels.sizes().reverse().transpose());
  SARA_DEBUG << "stacked kernels =\n"
             << kernels.matrix().transpose().format(HeavyFmt) << std::endl;

  SARA_DEBUG << "Copying the stacked kernels to CUDA constant memory"
             << std::endl;
  shakti::tic();
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(scg::kernels,  //
                                           kernels.data(),
                                           kernels.size() * sizeof(float)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      scg::kernel_count, kernels.sizes().data(), sizeof(int)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      scg::kernel_radius_max, kernels.sizes().data() + 1, sizeof(int)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(scg::kernel_radii,  //
                                           kernel_radii.data(),
                                           kernel_radii.size() * sizeof(int)));
  shakti::toc("copy to constant memory");

// #define THIS_WORKS
#ifdef THIS_WORKS
  auto kernel_count_copied = int{};
  auto kernel_radius_max_copied = int{};
  auto kernel_radii_copied = std::vector<int>(kernel_radii.size());
  auto kernels_copied = sara::Tensor_<float, 2>{kernels.sizes()};
  kernels_copied.flat_array().fill(-1);
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(
      kernels_copied.data(), scg::kernels, kernels.size() * sizeof(float)));
  SARA_DEBUG << "kernels copied=\n"
             << kernels_copied.matrix().transpose().format(HeavyFmt)
             << std::endl;

  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(&kernel_count_copied,  //
                                             scg::kernel_count, sizeof(int)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(&kernel_radius_max_copied,  //
                                             scg::kernel_radius_max,
                                             sizeof(int)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(  //
      kernel_radii_copied.data(),              //
      scg::kernel_radii,                       //
      kernel_count_copied * sizeof(int)));

  SARA_CHECK(kernel_count_copied);
  SARA_CHECK(kernel_radius_max_copied);
  SARA_CHECK(Eigen::Map<const Eigen::RowVectorXi>(kernel_radii_copied.data(),
                                                  kernel_radii_copied.size()));
  SARA_CHECK(kernels_copied.matrix().transpose());
#endif

  const auto w = 2 * 3840;
  const auto h = 2 * 2160;

  // Initialize the octave CUDA surface.
  auto h_arr = sara::Image<float, 2>{w, h};
  h_arr.flat_array().fill(0);
  h_arr(w / 2, h / 2) = 1.f;

  shakti::tic();
  auto d_in = shakti::MultiArray<float, 2>{h_arr.data(), {w, h}};
  shakti::toc("Copy from host memory");
  SARA_CHECK(d_in.padded_width());

  auto d_convx = shakti::MultiArray<float, 2>{{w, h}};
  auto d_convy = shakti::MultiArray<float, 2>{{w, h}};


  auto total_compute_time = double{};
  auto elapsed = double{};
  auto timer = shakti::Timer{};

  SARA_CHECK(scales.size());
  for (auto kernel_index = 0; kernel_index < scales.size(); ++kernel_index)
  {
    auto gauss_compute_time = double{};
    timer.restart();
    {
      const auto threadsperBlock = dim3(scg::kernel_max_radius, scg::tile_size);
      const auto numBlocks = dim3(
          (d_in.padded_width() + threadsperBlock.x - 1) / threadsperBlock.x,
          (d_in.height() + threadsperBlock.y - 1) / threadsperBlock.y);

      // x-convolution.
      scg::convx<<<numBlocks, threadsperBlock>>>(d_in.data(),     //
                                                 d_convx.data(),  //
                                                 d_in.width(),
                                                 d_in.height(),        //
                                                 d_in.padded_width(),  //
                                                 kernel_index);
    }
    elapsed = timer.elapsed_ms();
    gauss_compute_time += elapsed;
    SARA_DEBUG << sara::format("[x-conv][s=%d] %0.3f ms", kernel_index, elapsed)
               << std::endl;

    timer.restart();
    {
      const auto threadsperBlock = dim3(scg::tile_size, scg::kernel_max_radius);
      const auto numBlocks = dim3(
          (d_in.padded_width() + threadsperBlock.x - 1) / threadsperBlock.x,
          (d_in.height() + threadsperBlock.y - 1) / threadsperBlock.y);

      // y-convolution.
      scg::convy<<<numBlocks, threadsperBlock>>>(d_convx.data(),  //
                                                 d_convy.data(),  //
                                                 d_convx.width(),
                                                 d_convx.height(),        //
                                                 d_convx.padded_width(),  //
                                                 kernel_index);
    }
    elapsed = timer.elapsed_ms();
    gauss_compute_time += elapsed;
    SARA_DEBUG << sara::format("[y-conv][s=%d] %0.3f ms", kernel_index, elapsed)
               << std::endl;

    SARA_DEBUG << sara::format("[gauss-conv][s=%d] %0.3f ms", kernel_index,
                               gauss_compute_time)
               << std::endl;

    total_compute_time += gauss_compute_time;
  }
  SARA_DEBUG << sara::format("[gauss-octave][%4dx%4dx%u] %0.3f ms",  //
                             w, h, scales.size(), total_compute_time)
             << std::endl;


  auto h_convx = h_arr;
  auto h_convy = h_arr;
  d_convx.copy_to_host(h_convx.data());
  d_convy.copy_to_host(h_convy.data());

  if (w < 9 && h < 9)
  {
    std::cout << h_convx.matrix().format(HeavyFmt) << std::endl;
    std::cout << std::endl;
    std::cout << h_convy.matrix().format(HeavyFmt) << std::endl;
  }
}

// THE TIMING on Titan Xp
//
// [gauss-octave][7680x4320x6] 33.810 ms
// [gauss-octave][3840x2160x6]  8.534 ms
// [gauss-octave][1920x1080x6]  2.200 ms
// [gauss-octave][ 960x 540x6]  0.621 ms
// [gauss-octave][ 480x 270x6]  0.215 ms
// [gauss-octave][ 240x 135x6]  0.114 ms
