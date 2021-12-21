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
#endif


namespace DO::Sara {

  auto apply_gaussian_filter(const ImageView<float>& src, ImageView<float>& dst,
                             float sigma, float gauss_truncate) -> void
  {
    if (src.sizes() != dst.sizes())
      throw std::domain_error{
          "Source and destination image sizes are not equal!"};

#ifdef DO_SARA_USE_HALIDE
#  ifdef PROFILE_ME
    auto timer = Timer{};
    timer.restart();
#  endif
    auto src_buffer = Shakti::Halide::as_runtime_buffer_4d(src);
    auto dst_buffer = Shakti::Halide::as_runtime_buffer_4d(dst);
    shakti_gaussian_convolution_cpu(src_buffer, sigma, gauss_truncate,
                                    dst_buffer);
#  ifdef PROFILE_ME
    const auto elapsed = timer.elapsed_ms();
    SARA_DEBUG << "[CPU Halide Gaussian][" << src.sizes().transpose() << "] "
               << elapsed << " ms" << std::endl;
#  endif
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

}  // namespace DO::Sara
