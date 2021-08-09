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

#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/MultiArray/Offset.hpp>


namespace DO { namespace Shakti {

  __constant__ int histogram_padded_width;
  __constant__ Vector2i image_sizes;
  __constant__ Vector3i histogram_sizes;
  __constant__ Vector3i quantization_steps;
  __constant__ int num_pixels;

  __global__
  void init_color_histogram(float *histogram)
  {
    const auto i = offset<3>();
    histogram[i] = 0;
  }

  __global__
  void update_color_histogram(float *histogram, const Vector4ub *image)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();

    if (p.x() >= image_sizes[0] || p.y() >= image_sizes[1])
      return;

    const auto color = image[i];
    Vector3i c(color[0] / quantization_steps[0],
               color[1] / quantization_steps[1],
               color[2] / quantization_steps[2]);

    auto hist_idx = c[0]
                  + c[1] * histogram_padded_width
                  + c[2] * histogram_padded_width * histogram_sizes[1];
    atomicAdd(&histogram[hist_idx], 1.f);
  }

  MultiArray<float, 3>
  compute_color_histogram(const MultiArray<Vector4ub, 2>& image,
                          const Vector3i& quantization_steps)
  {
    const auto histogram_sizes =
        Vector3i(256 / quantization_steps[0], 256 / quantization_steps[1],
                 256 / quantization_steps[2]);

    MultiArray<float, 3> histogram{histogram_sizes};

    const auto image_sizes = image.sizes();
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(Shakti::image_sizes, &image_sizes,
                                             sizeof(Vector2i)));

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
        Shakti::histogram_sizes, &histogram_sizes, sizeof(Vector3i)));

    const auto histogram_padded_width = histogram.padded_width();
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
        Shakti::histogram_padded_width, &histogram_padded_width, sizeof(int)));

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
        Shakti::quantization_steps, &quantization_steps, sizeof(Vector3i)));

    auto block_sizes = dim3{};
    auto grid_sizes = dim3{};

    block_sizes = default_block_size_3d();
    grid_sizes = grid_size_3d(histogram);
    init_color_histogram<<<grid_sizes, block_sizes>>>(histogram.data());

    block_sizes = default_block_size_2d();
    grid_sizes = grid_size_2d(image);
    update_color_histogram<<<grid_sizes, block_sizes>>>(histogram.data(),
                                                        image.data());

    return histogram;
  }

  __global__
  void normalize_color_histogram(float *histogram)
  {
    const auto i = offset<3>();
    histogram[i] /= num_pixels;
  }

  void normalize_color_histogram(MultiArray<float, 3>& histogram, int num_pixels)
  {
    SHAKTI_SAFE_CUDA_CALL(
        cudaMemcpyToSymbol(Shakti::num_pixels, &num_pixels, sizeof(int)));

    const auto block_sizes = default_block_size_3d();
    const auto grid_sizes = grid_size_3d(histogram);
    normalize_color_histogram<<<grid_sizes, block_sizes>>>(histogram.data());
  }

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  DO_SHAKTI_EXPORT
  void compute_color_distribution(float *out_histogram,
                                  const Vector4ub *in_image,
                                  const int *in_image_sizes,
                                  const int *quantization_steps)
  {
    MultiArray<Vector4ub, 2> in_image_array{ in_image, in_image_sizes };
    MultiArray<float, 3> out_hist_array{
      compute_color_histogram(in_image_array, quantization_steps)
    };
    auto num_pixels = in_image_sizes[0] * in_image_sizes[1];
    normalize_color_histogram(out_hist_array, num_pixels);
    out_hist_array.copy_to_host(out_histogram);
  }

} /* namespace Shakti */
} /* namespace DO */
