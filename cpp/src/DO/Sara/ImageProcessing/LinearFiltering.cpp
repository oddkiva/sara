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

#  ifdef IMPL_V1
#    include "shakti_gaussian_convolution_cpu.h"
#  else
#    include "shakti_separable_convolution_2d_cpu.h"
#  endif
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

#  ifdef IMPL_V1
    shakti_gaussian_convolution_cpu(src_buffer, sigma, gauss_truncate,
                                    dst_buffer);
#  else
    const auto kernel = make_gaussian_kernel(sigma, gauss_truncate);
    auto kernel_buffer = Shakti::Halide::as_runtime_buffer(kernel);
    const auto kernel_size = static_cast<std::int32_t>(kernel.size());
    shakti_separable_convolution_2d_cpu(src_buffer, kernel_buffer, kernel_size,
                                        -kernel_size / 2, dst_buffer);
#  endif

#  ifdef PROFILE_ME
    const auto elapsed = timer.elapsed_ms();
    SARA_DEBUG << "[CPU Halide Gaussian][" << src.sizes().transpose() << "] "
               << elapsed << " ms" << std::endl;
    auto src_buffer = DO::Shakti::Halide::as_runtime_buffer_4d(src);
    auto kernel_buffer = DO::Shakti::Halide::as_runtime_buffer(kernel);
    auto dst_buffer = DO::Shakti::Halide::as_runtime_buffer_4d(dst);
    shakti_separable_convolution_2d_cpu(src_buffer, kernel_buffer, kernel_size,
                                        -center, dst_buffer);
#  endif
#else
    const auto kernel = make_gaussian_kernel(sigma, gauss_truncate);

    apply_row_based_filter(src, dst, kernel.data(), kernel.size());
    apply_column_based_filter(dst, dst, kernel.data(), kernel.size());
#endif
  }

}  // namespace DO::Sara
