// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>

#ifdef DO_SARA_USE_HALIDE
#  include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#  include "shakti_gaussian_convolution_cpu.h"
#  include "shakti_gaussian_convolution_gpu.h"
#  include "shakti_subtract_32f_cpu.h"
#endif


namespace DO::Sara {

  auto apply_gaussian_filter(const ImageView<float>& src, ImageView<float>& dst,
                             float sigma, float gauss_truncate) -> void
  {
    if (src.sizes() != dst.sizes())
      throw std::domain_error{
          "Source and destination image sizes are not equal!"};

#ifdef DO_SARA_USE_HALIDE
    auto timer = Timer{};
    timer.restart();
    auto src_buffer = Shakti::Halide::as_runtime_buffer_4d(src);
    auto dst_buffer = Shakti::Halide::as_runtime_buffer_4d(dst);
    shakti_gaussian_convolution_cpu(src_buffer, sigma, gauss_truncate,
                                    dst_buffer);
    const auto elapsed = timer.elapsed_ms();
    SARA_DEBUG << "[CPU Halide Gaussian][" << src.sizes().transpose() << "] "
               << elapsed << " ms" << std::endl;
#else
    // Compute the size of the Gaussian kernel.
    auto kernel_size = int(2 * gauss_truncate * sigma + 1);
    // Make sure the Gaussian kernel is at least of size 3 and is of odd size.
    kernel_size = std::max(3, kernel_size);
    if (kernel_size % 2 == 0)
      ++kernel_size;

    // Create the 1D Gaussian kernel.
    //
    // 1. Compute the value of the unnormalized Gaussian.
    const auto center = kernel_size / 2;
    auto kernel = std::vector<float>(kernel_size);
    for (int i = 0; i < kernel_size; ++i)
    {
      auto x = static_cast<float>(i - center);
      kernel[i] = exp(-square(x) / (2 * square(sigma)));
    }
    // 2. Calculate the normalizing factor.
    const auto sum_inverse =
        1 / std::accumulate(kernel.begin(), kernel.end(), 0.f);

    // Normalize the kernel.
    std::for_each(kernel.begin(), kernel.end(),
                  [sum_inverse](float& v) { v *= sum_inverse; });

    apply_row_based_filter(src, dst, &kernel[0], kernel_size);
    apply_column_based_filter(dst, dst, &kernel[0], kernel_size);

    apply_row_based_filter(src, dst, &kernel[0], kernel_size);
    apply_column_based_filter(dst, dst, &kernel[0], kernel_size);
#endif
  }

  auto difference_of_gaussians_pyramid(const ImagePyramid<float>& gaussians)
      -> ImagePyramid<float>
  {
    auto D = ImagePyramid<float>{};
    D.reset(gaussians.num_octaves(), gaussians.num_scales_per_octave() - 1,
            gaussians.scale_initial(), gaussians.scale_geometric_factor());

    for (auto o = 0; o < D.num_octaves(); ++o)
    {
      D.octave_scaling_factor(o) = gaussians.octave_scaling_factor(o);
      for (auto s = 0; s < D.num_scales_per_octave(); ++s)
      {
        D(s, o).resize(gaussians(s, o).sizes());

#ifdef DO_SARA_USE_HALIDE
        const auto& a = gaussians(s + 1, o);
        const auto& b = gaussians(s, o);
        auto& out = D(s, o);

        auto a_tensor_view = tensor_view(a).reshape(
            Eigen::Vector4i{1, 1, a.height(), a.width()});
        auto b_tensor_view = tensor_view(b).reshape(
            Eigen::Vector4i{1, 1, b.height(), b.width()});
        auto out_tensor_view = tensor_view(out).reshape(
            Eigen::Vector4i{1, 1, out.height(), out.width()});

        auto a_buffer = Shakti::Halide::as_runtime_buffer(a_tensor_view);
        auto b_buffer = Shakti::Halide::as_runtime_buffer(b_tensor_view);
        auto out_buffer = Shakti::Halide::as_runtime_buffer(out_tensor_view);

        shakti_subtract_32f_cpu(a_buffer, b_buffer, out_buffer);
#else
        throw std::runtime_error{"Not Implemented!"};
#endif
      }
    }
    return D;
  }

}  // namespace DO::Sara
