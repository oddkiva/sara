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

#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Shakti/Cuda/FeatureDetectors/GaussianOctaveKernels.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/Octave.hpp>
#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/Utilities/Timer.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace sc = DO::Shakti::Cuda;


namespace DO::Shakti::Cuda::Gaussian {

  static constexpr auto kernel_max_radius = 24;
  static constexpr auto tile_size = 1024 / kernel_max_radius;
  static constexpr auto sdata_rows = tile_size + 2 * kernel_max_radius;

  static constexpr auto kernel_count_max = 16;
  static constexpr auto kernel_capacity = kernel_max_radius * kernel_count_max;

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
    const auto& r = kernel_radii[kernel_index];

    if (x > w + r || y >= h)
      return;

    const auto gi = y * pitch + x;
    const auto ti = threadIdx.y * sdata_rows + threadIdx.x;

    const auto a = x - r;
    const auto b = x + r;

    // 1. Accumulate the shared data.
    if (a < 0)
      sdata[ti] = in[y * pitch];
    else if (a < w - 1)
      sdata[ti] = in[gi - r];
    else // if (a <= w + r)
      sdata[ti] = in[y * pitch + w - 1];

    if (b < 0)
      sdata[ti + 2 * r] = in[y * pitch];
    else if (b < w - 1)
      sdata[ti + 2 * r] = in[gi + r];
    else // if (b <= w + r)
      sdata[ti + 2 * r] = in[y * pitch + w - 1];
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
    const int& r = kernel_radii[kernel_index];
    if (x >= w || y > h + r)
      return;

    const int gi = y * pitch + x;

    // 1. Accumulate the shared data.
    const auto& tx = threadIdx.x;
    const auto& ty = threadIdx.y;
    const auto t = tx * sdata_rows + ty;

    const auto a = y - r;
    const auto b = y + r;

    if (a < 0)
      sdata[t] = in[x];
    else if (a < h - 1)
      sdata[t] = in[gi - r * pitch];
    else // if (a <= h + 2 * r)
      sdata[t] = in[(h - 1) * pitch + x];

    if (b < 0)
      sdata[t + 2 * r] = in[x];
    else if (b < h - 1)
      sdata[t + 2 * r] = in[gi + r * pitch];
    else // if (b <= h + r)
      sdata[t + 2 * r] = in[(h - 1) * pitch + x];
    __syncthreads();

    if (y >= h)
      return;

    // Pick the right kernel.
    const auto kernel = kernels + kernel_index * kernel_radius_max;

    // 2. Convolve.
    auto val = sdata[t + r] * kernel[0];
#pragma unroll
    for (auto k = 1; k <= r; ++k)
      val += kernel[k] * (sdata[t + r - k] + sdata[t + r + k]);
    out[gi] = val;
  }

  inline auto copy_gaussian_kernels_to_constant_memory(
      const Shakti::Cuda::GaussianOctaveKernels<float>& gok) -> void
  {
    SARA_DEBUG << "Copying the stacked kernels to CUDA constant memory"
               << std::endl;
    shakti::tic();
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(  //
        kernels,                               //
        gok.kernels.data(),                    //
        gok.kernels.size() * sizeof(float)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(  //
        kernel_count,                          //
        gok.kernels.sizes().data(),            //
        sizeof(int)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(  //
        kernel_radius_max,                     //
        gok.kernels.sizes().data() + 1,        //
        sizeof(int)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(  //
        kernel_radii,                          //
        gok.kernel_radii.data(),               //
        gok.kernel_radii.size() * sizeof(int)));
    shakti::toc("copy to constant memory");
  }

  inline auto peek_gaussian_kernels_in_constant_memory()
  {
    const auto HeavyFmt =
        Eigen::IOFormat(3, 0, ", ", ",\n", "[", "]", "[", "]");

    auto kernel_count_copied = int{};
    auto kernel_radius_max_copied = int{};
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(&kernel_count_copied,  //
                                               kernel_count, sizeof(int)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(&kernel_radius_max_copied,  //
                                               kernel_radius_max, sizeof(int)));

    auto kernels_copied = sara::Tensor_<float, 2>{
        kernel_count_copied,      //
        kernel_radius_max_copied  //
    };
    auto kernel_radii_copied = std::vector<int>(kernel_count_copied);

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(
        kernels_copied.data(), kernels, kernels_copied.size() * sizeof(float)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(  //
        kernel_radii_copied.data(),              //
        kernel_radii,                            //
        kernel_count_copied * sizeof(int)));

    SARA_DEBUG << "kernels copied=\n"
               << kernels_copied.matrix().transpose().format(HeavyFmt)
               << std::endl;
    SARA_DEBUG << "kernel radii =\n"
               << Eigen::Map<const Eigen::RowVectorXi>(
                      kernel_radii_copied.data(), kernel_radii_copied.size())
               << std::endl;

    SARA_CHECK(kernel_count_copied);
    SARA_CHECK(kernel_radius_max_copied);
  }

}  // namespace DO::Shakti::Cuda::Gaussian

namespace scg = sc::Gaussian;


BOOST_AUTO_TEST_CASE(test_convolve)
{
  auto gok = sc::GaussianOctaveKernels<float>{};
  scg::copy_gaussian_kernels_to_constant_memory(gok);
  scg::peek_gaussian_kernels_in_constant_memory();

// #define CHECK_IMPL
#ifdef CHECK_IMPL
  auto w = 5;
  auto h = 5;
#else
  auto w = 2 * 3840;
  auto h = 2 * 2160;
#endif

  // Timing stuff.
  auto pyramid_compute_time = double{};
  auto elapsed = double{};
  auto timer = shakti::Timer{};

  do
  {
    // Initialize the octave CUDA surface.
    auto h_arr = sara::Image<float, 2>{w, h};
    h_arr.flat_array().fill(0);
    h_arr(w / 2, h / 2) = 1.f;

    // shakti::tic();
    auto d_in = shakti::MultiArray<float, 2>{h_arr.data(), {w, h}};
    // shakti::toc("Copy from host memory");

    // shakti::tic();
    auto d_convx = shakti::MultiArray<float, 2>{{w, h}};
    auto d_convy = shakti::MultiArray<float, 2>{{w, h}};
    // shakti::toc("Resize device buffers");

    auto octave_compute_time = double{};
#ifdef CHECK_IMPL
    const auto kernel_index = 0;
#else  // BENCHMARK
    for (auto kernel_index = 0; kernel_index < gok.scales.size();
         ++kernel_index)
#endif
    {
      auto gauss_compute_time = double{};
      timer.restart();
      {
        const auto threadsperBlock =
            dim3(scg::kernel_max_radius, scg::tile_size);
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
      SARA_DEBUG << sara::format("[x-conv][s=%d] %0.3f ms", kernel_index,
                                 elapsed)
                 << std::endl;

      timer.restart();
      {
        const auto threadsperBlock =
            dim3(scg::tile_size, scg::kernel_max_radius);
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
      SARA_DEBUG << sara::format("[y-conv][s=%d] %0.3f ms", kernel_index,
                                 elapsed)
                 << std::endl;

      SARA_DEBUG << sara::format("[gaussf][s=%d] %0.3f ms", kernel_index,
                                 gauss_compute_time)
                 << std::endl;

      octave_compute_time += gauss_compute_time;

      if (w < 9 && h < 9)
      {
        auto h_convx = h_arr;
        auto h_convy = h_arr;
        d_convx.copy_to_host(h_convx.data());
        d_convy.copy_to_host(h_convy.data());

        const auto HeavyFmt =
            Eigen::IOFormat(4, 0, ", ", ",\n", "[", "]", "[", "]");

        std::cout << h_convx.matrix().format(HeavyFmt) << std::endl;
        std::cout << std::endl;
        std::cout << h_convy.matrix().format(HeavyFmt) << std::endl;
      }
    }
    SARA_DEBUG << sara::format("[octave][%4dx%4dx%u] %0.3f ms",  //
                               w, h, gok.scales.size(), octave_compute_time)
               << std::endl;

    pyramid_compute_time += octave_compute_time;

    w /= 2;
    h /= 2;

  } while (w >= 32 && h >= 16);

  SARA_DEBUG << sara::format("[pyramid][%4dx%4dx%u] %0.3f ms",  //
                             w, h, gok.scales.size(), pyramid_compute_time)
             << std::endl;

  SARA_CHECK(scg::kernel_max_radius);
  SARA_CHECK(scg::tile_size);
}

// THE TIMING on Titan 1080 Xp
//
// No memory error according to the memcheck tool:
// $> compute-sanitizer --tool memcheck ./bin/shakti_test_gaussian_convolution
//
// [gauss-octave][7680x4320x6] 21.212 ms
// [gauss-octave][3840x2160x6]  5.384 ms
// [gauss-octave][1920x1080x6]  1.416 ms
// [gauss-octave][ 960x 540x6]  0.436 ms
// [gauss-octave][ 480x 270x6]  0.151 ms
// [gauss-octave][ 240x 135x6]  0.091 ms
//
// 4K Gaussian pyramid = 7.797 ms
// 8K Gaussian pyramid = 29.538 ms
