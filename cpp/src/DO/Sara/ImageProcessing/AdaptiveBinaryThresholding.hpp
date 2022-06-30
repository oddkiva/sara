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

  template <typename T>
  auto adaptive_thresholding(const ImageView<T>& src,       //
                             const ImageView<T>& kernel,    //
                             ImageView<std::uint8_t>& dst,  //
                             const T tolerance_param) -> void
  {
    const Eigen::Vector2i c = kernel.sizes() / 2;
    const Eigen::Vector2i r = kernel.sizes() / 2;

    const Eigen::Vector2i a = Eigen::Vector2i::Zero();
    const Eigen::Vector2i b = (src.sizes().array() - 1).matrix();
    const auto clamp = [&a, &b](const Eigen::Vector2i& x) -> Eigen::Vector2i {
      return x.cwiseMax(a).cwiseMin(b);
    };

    const auto sum = kernel.flat_array().sum();

#pragma omp parallel for collapse(2)
    for (auto y = 0; y < src.height(); ++y)
    {
      for (auto x = 0; x < src.width(); ++x)
      {
        auto mean = T{};
        for (auto v = -r.y(); v <= r.y(); ++v)
        {
          for (auto u = -r.x(); u <= r.x(); ++u)
          {
            const auto uv = Eigen::Vector2i{r.x() + u, r.y() + v};
            const auto p1 = clamp(Eigen::Vector2i{x + u, y + v});

            mean += src(p1) * kernel(uv);
          }
        }
        mean /= sum;

        dst(x, y) = src(x, y) > (mean - tolerance_param) ? 1 : 0;
      }
    }
  }

  auto gaussian_adaptive_threshold(const ImageView<float>& src,
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
