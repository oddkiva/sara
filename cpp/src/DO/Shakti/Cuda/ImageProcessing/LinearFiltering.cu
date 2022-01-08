// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Shakti/Cuda/ImageProcessing.hpp>
#include <DO/Shakti/Cuda/ImageProcessing/Kernels/Convolution.hpp>

#include <DO/Shakti/Cuda/Utilities.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <algorithm>


namespace DO { namespace Shakti {

  void apply_row_based_convolution(float* out, const float* in,
                                   const float* kernel, int kernel_size,
                                   const int* sizes)
  {
    TextureArray<float> in_array{in, sizes};
    MultiArray<float, 2> out_array{sizes};

    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(out_array);

    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float_texture, in_array));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(Shakti::kernel, kernel,
                                             sizeof(float) * kernel_size));
    SHAKTI_SAFE_CUDA_CALL(
        cudaMemcpyToSymbol(Shakti::kernel_size, &kernel_size, sizeof(int)));
    SHAKTI_SAFE_CUDA_CALL(
        cudaMemcpyToSymbol(Shakti::image_sizes, sizes, sizeof(int2)));

    apply_row_based_convolution<<<grid_size, block_size>>>(out_array.data());
    SHAKTI_CHECK_LAST_ERROR();

    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float_texture));

    out_array.copy_to_host(out);
  }

  void apply_column_based_convolution(float* out, const float* in,
                                      const float* kernel, int kernel_size,
                                      const int* sizes)
  {
    TextureArray<float> in_array{in, sizes};
    MultiArray<float, 2> out_array{sizes};

    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(out_array);

    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float_texture, in_array));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(Shakti::kernel, kernel,
                                             sizeof(float) * kernel_size));
    SHAKTI_SAFE_CUDA_CALL(
        cudaMemcpyToSymbol(Shakti::kernel_size, &kernel_size, sizeof(int)));
    SHAKTI_SAFE_CUDA_CALL(
        cudaMemcpyToSymbol(Shakti::image_sizes, sizes, sizeof(int2)));

    apply_column_based_convolution<<<grid_size, block_size>>>(out_array.data());
    SHAKTI_CHECK_LAST_ERROR();

    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float_texture));

    out_array.copy_to_host(out);
  }

}}  // namespace DO::Shakti


namespace DO { namespace Shakti {

  void GaussianFilter::set_sigma(float sigma)
  {
    auto kernel_size = static_cast<int>(2.f * _truncation_factor * sigma + 1.f);
    kernel_size = std::max(3, kernel_size);
    if (kernel_size % 2 == 0)
      ++kernel_size;

    auto sum = float{0.f};
    _kernel.resize(kernel_size);
    for (auto i = int{0}; i < kernel_size; ++i)
    {
      auto x = i - kernel_size / 2;
      _kernel[i] = exp(-x * x / (2.f * sigma * sigma));
      sum += _kernel[i];
    }
    for (auto i = int{0}; i < kernel_size; ++i)
      _kernel[i] /= sum;

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(Shakti::kernel, _kernel.data(),
                                             sizeof(float) * _kernel.size()));
    SHAKTI_SAFE_CUDA_CALL(
        cudaMemcpyToSymbol(Shakti::kernel_size, &kernel_size, sizeof(int)));
  }

  void GaussianFilter::operator()(float* out, const float* in,
                                  const int* sizes) const
  {
    auto timer = Shakti::Timer{};
    auto elapsed = double{};

    timer.restart();
    TextureArray<float> in_array{
        in, {sizes[0], sizes[1]}, sizes[0] * sizeof(float)};
    elapsed = timer.elapsed_ms();
    std::cout << "[[GF] init GPU texture] " << elapsed << " ms" << std::endl;

    timer.restart();
    MultiArray<float, 2> out_array{{sizes[0], sizes[1]}};
    elapsed = timer.elapsed_ms();
    std::cout << "[[GF] init out device memory] " << elapsed << " ms" << std::endl;

    timer.restart();
    SHAKTI_SAFE_CUDA_CALL(
        cudaMemcpyToSymbol(Shakti::image_sizes, sizes, sizeof(int2)));
    elapsed = timer.elapsed_ms();

    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(out_array, block_size);
    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float_texture, in_array));
    std::cout << "[[GF] init Gaussian filter] " << elapsed << " ms" << std::endl;

    timer.restart();
    apply_column_based_convolution<<<grid_size, block_size>>>(out_array.data());
    SHAKTI_CHECK_LAST_ERROR();
    elapsed = timer.elapsed_ms();
    std::cout << "[[GF] apply X-Gaussian filter] " << elapsed << " ms"
              << std::endl;

    timer.restart();
    in_array.copy_from(out_array.data(), out_array.sizes(), out_array.pitch(),
                       cudaMemcpyDeviceToDevice);
    elapsed = timer.elapsed_ms();
    std::cout << "[[GF] copy device-to-device] " << elapsed << " ms"
              << std::endl;

    timer.restart();
    apply_row_based_convolution<<<grid_size, block_size>>>(out_array.data());
    SHAKTI_CHECK_LAST_ERROR();
    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float_texture));
    elapsed = timer.elapsed_ms();
    std::cout << "[[GF] apply Y-Gaussian filter] " << elapsed << " ms"
              << std::endl;

    timer.restart();
    out_array.copy_to_host(out);
    elapsed = timer.elapsed_ms();
    std::cout << "[[GF] copy to host] " << elapsed << " ms" << std::endl;
  }

  MultiArray<float, 2> GaussianFilter::operator()(TextureArray<float>& in) const
  {
    MultiArray<float, 2> out{in.sizes()};

    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(out);

    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float_texture, in));
    {
      apply_column_based_convolution<<<grid_size, block_size>>>(out.data());
      in.copy_from(out.data(), out.sizes(), out.pitch(),
                   cudaMemcpyDeviceToDevice);
      apply_row_based_convolution<<<grid_size, block_size>>>(out.data());
    }
    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float_texture));

    return out;
  }

}}  // namespace DO::Shakti
