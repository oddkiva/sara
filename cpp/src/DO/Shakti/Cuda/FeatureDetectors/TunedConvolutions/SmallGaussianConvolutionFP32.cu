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

#include <DO/Shakti/Cuda/Utilities/ErrorCheck.hpp>
#include <DO/Shakti/Cuda/Utilities/StringFormat.hpp>
#include <DO/Shakti/Cuda/Utilities/Timer.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/TunedConvolutions/SmallGaussianConvolutionFP32.hpp>


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
    const int& r = kernel_radii[kernel_index];

    if (x >= max(w, r) + r || y >= h)
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
    else  // if (a <= w + r)
      sdata[ti] = in[y * pitch + w - 1];

    if (b < 0)
      sdata[ti + 2 * r] = in[y * pitch];
    else if (b < w - 1)
      sdata[ti + 2 * r] = in[gi + r];
    else  // if (b <= w + r)
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
    if (x >= w || y > max(h, r) + r)
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
    else  // if (a <= h + 2 * r)
      sdata[t] = in[(h - 1) * pitch + x];

    if (b < 0)
      sdata[t + 2 * r] = in[x];
    else if (b < h - 1)
      sdata[t + 2 * r] = in[gi + r * pitch];
    else  // if (b <= h + r)
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

  __global__ auto convx_at_scale(cudaSurfaceObject_t in_3d, float* out_2d,
                                 int scale, int w, int h, int out_pitch) -> void
  {
    __shared__ float sdata[tile_size * sdata_rows];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int& r = kernel_radii[scale];
    const auto scale_prev = scale - 1;

    if (x >= max(w, r) + r || y >= h)
      return;

    const auto ti = threadIdx.y * sdata_rows + threadIdx.x;

    const auto a = x - r;
    const auto b = x + r;

    // 1. Accumulate the shared data.
    surf2DLayeredread(&sdata[ti],                        //
                      in_3d,                             //
                      a * sizeof(float), y, scale_prev,  //
                      cudaBoundaryModeClamp);
    surf2DLayeredread(&sdata[ti + 2 * r],                //
                      in_3d,                             //
                      b * sizeof(float), y, scale_prev,  //
                      cudaBoundaryModeClamp);
    __syncthreads();


    // Pick the right kernel.
    const auto kernel = kernels + scale * kernel_radius_max;

    // 2. Convolve.
    auto val = kernel[0] * sdata[ti + r];
#pragma unroll
    for (auto k = 1; k <= r; ++k)
      val += kernel[k] * (sdata[ti + r - k] + sdata[ti + r + k]);

    const auto gi = y * out_pitch + x;
    out_2d[gi] = val;
  }

  __global__ auto convy_at_scale(const float* in_2d,
                                 cudaSurfaceObject_t out_3d,  //
                                 int s,                       //
                                 int w, int h, int in_pitch) -> void
  {
    __shared__ float sdata[tile_size * sdata_rows];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int& r = kernel_radii[0];
    if (x >= w || y > max(h, r) + r)
      return;

    const int gi = y * in_pitch + x;

    // 1. Accumulate the shared data.
    const auto& tx = threadIdx.x;
    const auto& ty = threadIdx.y;
    const auto t = tx * sdata_rows + ty;

    const auto a = y - r;
    const auto b = y + r;

    if (a < 0)
      sdata[t] = in_2d[x];
    else if (a < h - 1)
      sdata[t] = in_2d[gi - r * in_pitch];
    else  // if (a <= h + 2 * r)
      sdata[t] = in_2d[(h - 1) * in_pitch + x];

    if (b < 0)
      sdata[t + 2 * r] = in_2d[x];
    else if (b < h - 1)
      sdata[t + 2 * r] = in_2d[gi + r * in_pitch];
    else  // if (b <= h + r)
      sdata[t + 2 * r] = in_2d[(h - 1) * in_pitch + x];
    __syncthreads();

    if (y >= h)
      return;

    // Pick the right kernel.
    const auto kernel = kernels;

    // 2. Convolve.
    auto val = sdata[t + r] * kernel[0];
#pragma unroll
    for (auto k = 1; k <= r; ++k)
      val += kernel[k] * (sdata[t + r - k] + sdata[t + r + k]);

    surf2DLayeredwrite(val, out_3d, x * sizeof(float), y, s);
  }

  auto DeviceGaussianFilterBank::copy_filters_to_device_constant_memory()
      -> void
  {
    SARA_DEBUG << "Copying the stacked kernels to CUDA constant memory"
               << std::endl;
    auto timer = Timer{};
    tic(timer);
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(  //
        kernels,                               //
        _filter_bank.kernels.data(),           //
        _filter_bank.kernels.size() * sizeof(float)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(  //
        kernel_count,                          //
        _filter_bank.kernels.sizes().data(),   //
        sizeof(int)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(     //
        kernel_radius_max,                        //
        _filter_bank.kernels.sizes().data() + 1,  //
        sizeof(int)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(  //
        kernel_radii,                          //
        _filter_bank.kernel_radii.data(),      //
        _filter_bank.kernel_radii.size() * sizeof(int)));
    toc(timer, "copy to constant memory");
  }

  auto DeviceGaussianFilterBank::peek_filters_in_device_constant_memory()
      -> void
  {
    const auto HeavyFmt =
        Eigen::IOFormat(3, 0, ", ", ",\n", "[", "]", "[", "]");

    auto kernel_count_copied = int{};
    auto kernel_radius_max_copied = int{};
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(&kernel_count_copied,  //
                                               kernel_count, sizeof(int)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(&kernel_radius_max_copied,  //
                                               kernel_radius_max, sizeof(int)));

    auto kernels_copied = Sara::Tensor_<float, 2>{
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

  auto DeviceGaussianFilterBank::operator()(
      const MultiArrayView<float, 2, RowMajorStrides>& d_in,
      MultiArrayView<float, 2, RowMajorStrides>& d_convx,
      MultiArrayView<float, 2, RowMajorStrides>& d_convy,
      int kernel_index) const -> void
  {
// #define PROFILE_GAUSSIAN_CONVOLUTION
#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    auto timer = Timer{};
    auto elapsed = double{};
    timer.restart();
#endif
    {
      const auto threadsperBlock = dim3(kernel_max_radius, tile_size);
      const auto numBlocks = dim3(
          (d_in.padded_width() + threadsperBlock.x - 1) / threadsperBlock.x,
          (d_in.height() + threadsperBlock.y - 1) / threadsperBlock.y);

      // x-convolution.
      convx<<<numBlocks, threadsperBlock>>>(d_in.data(),          //
                                            d_convx.data(),       //
                                            d_in.width(),         //
                                            d_in.height(),        //
                                            d_in.padded_width(),  //
                                            kernel_index);
    }
#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    elapsed = timer.elapsed_ms();
    SHAKTI_STDOUT << format("[x-conv][s=%d] %0.3f ms", kernel_index, elapsed)
                  << std::endl;
#endif

#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    timer.restart();
#endif
    {
      const auto threadsperBlock = dim3(tile_size, kernel_max_radius);
      const auto numBlocks = dim3(
          (d_in.padded_width() + threadsperBlock.x - 1) / threadsperBlock.x,
          (d_in.height() + threadsperBlock.y - 1) / threadsperBlock.y);

      // y-convolution.
      convy<<<numBlocks, threadsperBlock>>>(d_convx.data(),  //
                                            d_convy.data(),  //
                                            d_convx.width(),
                                            d_convx.height(),        //
                                            d_convx.padded_width(),  //
                                            kernel_index);
    }
#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    elapsed = timer.elapsed_ms();
    SHAKTI_STDOUT << format("[y-conv][s=%d] %0.3f ms", kernel_index, elapsed)
                  << std::endl;
#endif
  }

  auto DeviceGaussianFilterBank::operator()(
      const MultiArrayView<float, 2, RowMajorStrides>& d_in,
      MultiArrayView<float, 2, RowMajorStrides>& d_convx,
      Octave<float>& gaussian_octave) const -> void
  {
    if (!gaussian_octave.surface_object().initialized())
    {
      SHAKTI_STDOUT << "INIT GAUSSIAN OCTAVE SURFACE" << std::endl;
      gaussian_octave.init_surface();
    }

    compute_zero_scale(d_in, d_convx, gaussian_octave);
    for (auto s = 1; s < gaussian_octave.scale_count(); ++s)
      compute_nonzero_scale(gaussian_octave, d_convx, s);
  }

  auto DeviceGaussianFilterBank::compute_zero_scale(  //
      const MultiArrayView<float, 2, RowMajorStrides>& d_in,
      MultiArrayView<float, 2, RowMajorStrides>& d_convx,
      Octave<float>& gaussian_octave) const -> void
  {
// #define PROFILE_GAUSSIAN_CONVOLUTION
#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    auto timer = Timer{};
    auto elapsed = double{};
    timer.restart();
#endif
    {
      const auto threadsperBlock = dim3(kernel_max_radius, tile_size);
      const auto numBlocks = dim3(
          (d_in.padded_width() + threadsperBlock.x - 1) / threadsperBlock.x,
          (d_in.height() + threadsperBlock.y - 1) / threadsperBlock.y);

      // x-convolution.
      convx<<<numBlocks, threadsperBlock>>>(d_in.data(),          //
                                            d_convx.data(),       //
                                            d_in.width(),         //
                                            d_in.height(),        //
                                            d_in.padded_width(),  //
                                            0);
    }
#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    elapsed = timer.elapsed_ms();
    SHAKTI_STDOUT << format("[x-conv][s=%d] %0.3f ms", 0, elapsed) << std::endl;
#endif

#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    timer.restart();
#endif
    {
      const auto threadsperBlock = dim3(tile_size, kernel_max_radius);
      const auto numBlocks = dim3(
          (d_in.padded_width() + threadsperBlock.x - 1) / threadsperBlock.x,
          (d_in.height() + threadsperBlock.y - 1) / threadsperBlock.y);

      // y-convolution.
      convy_at_scale<<<numBlocks, threadsperBlock>>>(
          d_convx.data(),                    //
          gaussian_octave.surface_object(),  //
          0,                                 //
          d_convx.width(),                   //
          d_convx.height(),                  //
          d_convx.padded_width());
    }
#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    elapsed = timer.elapsed_ms();
    SHAKTI_STDOUT << format("[y-conv][s=%d] %0.3f ms", 0, elapsed) << std::endl;
#endif
  }

  auto DeviceGaussianFilterBank::compute_nonzero_scale(
      Octave<float>& gaussian_octave,
      MultiArrayView<float, 2, RowMajorStrides>& d_convx, int scale) const
      -> void
  {
#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    auto timer = Timer{};
    auto elapsed = double{};
    timer.restart();
#endif
    {
      const auto threadsperBlock = dim3(kernel_max_radius, tile_size);
      const auto numBlocks = dim3(
          (d_convx.padded_width() + threadsperBlock.x - 1) / threadsperBlock.x,
          (d_convx.height() + threadsperBlock.y - 1) / threadsperBlock.y);

      // x-convolution.
      convx_at_scale<<<numBlocks, threadsperBlock>>>(
          gaussian_octave.surface_object(),  //
          d_convx.data(),                    //
          scale,                             //
          d_convx.width(),                   //
          d_convx.height(),                  //
          d_convx.padded_width());
    }
#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    elapsed = timer.elapsed_ms();
    SHAKTI_STDOUT << format("[x-conv][s=%d] %0.3f ms", scale, elapsed)
                  << std::endl;
#endif

#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    timer.restart();
#endif
    {
      const auto threadsperBlock = dim3(tile_size, kernel_max_radius);
      const auto numBlocks = dim3(
          (d_convx.padded_width() + threadsperBlock.x - 1) / threadsperBlock.x,
          (d_convx.height() + threadsperBlock.y - 1) / threadsperBlock.y);

      // y-convolution.
      convy_at_scale<<<numBlocks, threadsperBlock>>>(
          d_convx.data(),                    //
          gaussian_octave.surface_object(),  //
          scale,                             //
          d_convx.width(),                   //
          d_convx.height(),                  //
          d_convx.padded_width());
    }
#ifdef PROFILE_GAUSSIAN_CONVOLUTION
    elapsed = timer.elapsed_ms();
    SHAKTI_STDOUT << format("[y-conv][s=%d] %0.3f ms", scale, elapsed)
                  << std::endl;
#endif
  }

}  // namespace DO::Shakti::Cuda::Gaussian
