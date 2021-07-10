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
#include <DO/Shakti/Cuda/ImageProcessing/Kernels/Differential.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>


namespace DO { namespace Shakti {

  MultiArray<Vector2f, 2> gradient(const TextureArray<float>& in)
  {
    MultiArray<Vector2f, 2> out{in.sizes()};

    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(out);

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(Shakti::image_sizes,
                                             in.sizes().data(), sizeof(int2)));

    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float_texture, in));
    apply_gradient_kernel<<<grid_size, block_size>>>(out.data());
    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float_texture));

    return out;
  }

  MultiArray<Vector2f, 2> gradient_polar_coords(const TextureArray<float>& in)
  {
    MultiArray<Vector2f, 2> out{in.sizes()};

    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(out);

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(Shakti::image_sizes,
                                             in.sizes().data(), sizeof(int2)));

    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float_texture, in));
    apply_gradient_polar_coordinates_kernel<<<grid_size, block_size>>>(
        out.data());
    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float_texture));

    return out;
  }

  MultiArray<float, 2> gradient_squared_norm(const TextureArray<float>& in)
  {
    MultiArray<float, 2> out{in.sizes()};

    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(out);

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(Shakti::image_sizes,
                                             in.sizes().data(), sizeof(int2)));

    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float_texture, in));
    apply_gradient_squared_norms_kernel<<<grid_size, block_size>>>(out.data());
    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float_texture));

    return out;
  }

  MultiArray<float, 2> squared_norm(const MultiArray<Vector2f, 2>& in)
  {
    MultiArray<float, 2> out{in.sizes()};

    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(out);

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(Shakti::image_sizes,
                                             in.sizes().data(), sizeof(int2)));

    apply_squared_norms_kernel<<<grid_size, block_size>>>(out.data(),
                                                          in.data());
    return out;
  }

  MultiArray<float, 2> laplacian(const TextureArray<float>& in)
  {
    MultiArray<float, 2> out{in.sizes()};

    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(out);

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(Shakti::image_sizes,
                                             in.sizes().data(), sizeof(int2)));

    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float_texture, in));
    apply_laplacian_kernel<<<grid_size, block_size>>>(out.data());
    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float_texture));

    return out;
  }

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  void compute_x_derivative(float *out, const float *in, const int *sizes)
  {
    const float kernel[] = {-1.f, 0.f, 1.f};
    const int kernel_size{3};
    apply_column_based_convolution(out, in, kernel, kernel_size, sizes);
  }

  void compute_y_derivative(float* out, const float* in, const int* sizes)
  {
    const float kernel[] = {-1.f, 0.f, 1.f};
    const int kernel_size{3};
    apply_row_based_convolution(out, in, kernel, kernel_size, sizes);
  }

  void compute_gradient(Vector2f* out, const float* in, const int* sizes)
  {
    TextureArray<float> in_cuda_array{in, {sizes[0], sizes[1]}};
    MultiArray<Vector2f, 2> gradients{gradient(in_cuda_array)};
    gradients.copy_to_host(out);
  }

  void compute_gradient_squared_norms(float* out, const float* in,
                                      const int* sizes)
  {
    TextureArray<float> in_cuda_array{in, {sizes[0], sizes[1]}};
    MultiArray<float, 2> gradient_squared_norms{
        gradient_squared_norm(in_cuda_array)};
    gradient_squared_norms.copy_to_host(out);
  }

  void compute_laplacian(float* out, const float* in, const int* sizes)
  {
    TextureArray<float> in_cuda_array{in, {sizes[0], sizes[1]}};
    MultiArray<float, 2> laplacians{laplacian(in_cuda_array)};
    laplacians.copy_to_host(out);
  }

} /* namespace Shakti */
} /* namespace DO */
