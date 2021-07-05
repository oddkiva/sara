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

#pragma once

#include <DO/Sara/Core/Image.hpp>


namespace DO::Sara {

  // Implemented from Wikipedia's explanation.
  inline auto otsu_adaptive_binarization(const ImageView<float>& image)
  {
    const auto image_quantized = image.convert<std::uint8_t>();

    // Calculate the discrete probability distribution function.
    Eigen::Array<float, 256, 1> pdf = Eigen::Array<float, 256, 1>::Zero();
    for (const auto& pixel : image_quantized)
      pdf(pixel) += 1;
    pdf /= pdf.sum();

    // Calculate the cumulated distribution function for the 0 class (i.e., the
    // black color class).
    auto cdf = Eigen::Array<float, 256, 1>{};
    std::partial_sum(pdf.data(), pdf.data() + pdf.size(), cdf.data());

    // The mean intensity value.
    const auto mean = [&pdf]() {
      auto m = 0.f;
      for (auto t = 0; t < pdf.size(); ++t)
        m += t * pdf(t);
      return m;
    }();

    auto t = 0;

    auto cdf_0 = cdf(0);
    auto cdf_1 = 1 - cdf_0;

    auto mean_0_unnormalized = 0;
    auto mean_1_unnormalized = mean;

    auto mean_0 = mean_0_unnormalized;
    auto mean_1 = mean;

    auto best_t = cdf_0 * cdf_1 * std::pow(mean_0 - mean_1, 2);
    auto best_score = cdf_0 * cdf_1 * std::pow(mean_0 - mean_1, 2);

    for (t = 1; t < 256; ++t)
    {
      cdf_0 = cdf(t);
      cdf_1 = 1 - cdf_0;

      mean_0_unnormalized += t * pdf(t);
      mean_1_unnormalized -= t * pdf(t);

      mean_0 = mean_0_unnormalized / cdf_0;
      mean_1 = mean_1_unnormalized / cdf_1;

      const auto score = cdf_0 * cdf_1 * std::pow(mean_0 - mean_1, 2);
      if (score > best_score)
      {
        best_t = t;
        best_score = score;
      }
    }

    auto binary = Image<std::uint8_t>{image.sizes()};
    std::transform(image_quantized.begin(), image_quantized.end(),
                   binary.begin(),
                   [best_t](const auto& v) { return v < best_t ? 0 : 255; });

    return binary;
  }

  // Implemented from Wikipedia's explanation.
  inline auto otsu_adaptive_binarization(const ImageView<float>& image,
                                         const Eigen::Vector2i& grid_sizes)
  {
    auto image_binarized = Image<std::uint8_t>{image.sizes()};
    for (auto y = 0; y < image.height(); y += grid_sizes.y())
    {
      for (auto x = 0; x < image.width(); x += grid_sizes.x())
      {
        const Eigen::Vector2i start = Eigen::Vector2i{x, y};
        const Eigen::Vector2i end = (start + grid_sizes).cwiseMin(image.sizes());
        const auto patch = safe_crop(image, start, end);
        const auto patch_binarized = otsu_adaptive_binarization(patch);

        for (auto v = 0; v < patch_binarized.height(); ++v)
          for (auto u = 0; u < patch_binarized.width(); ++u)
            image_binarized(x + u, y + v) = patch_binarized(u, v);
      }
    }
    return image_binarized;
  }

}  // namespace DO::Sara
