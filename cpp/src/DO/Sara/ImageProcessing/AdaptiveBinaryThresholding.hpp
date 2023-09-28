// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <algorithm>
#ifdef _WIN32
#  include <execution>
#endif

#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>


namespace DO::Sara {

  inline auto gaussian_adaptive_threshold(const ImageView<float>& src,
                                          const float sigma,
                                          const float gauss_truncate,
                                          const float tolerance_parameter,
                                          ImageView<std::uint8_t>& segmentation)
      -> void
  {
    const auto src_conv = gaussian(src, sigma, gauss_truncate);
    std::transform(
#ifdef _WIN32
        std::execution::par_unseq,
#endif
        src_conv.begin(), src_conv.end(), src.begin(), segmentation.begin(),
        [tolerance_parameter](const auto& mean, const auto& val) {
          return val > (mean - tolerance_parameter) ? 255 : 0;
        });
  };

}  // namespace DO::Sara
