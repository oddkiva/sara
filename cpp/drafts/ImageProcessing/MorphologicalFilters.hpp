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

  inline auto erode(const Image<std::uint8_t>& image, int radius)
  {
    auto image_eroded = Image<std::uint8_t>{image.sizes()};

    for (auto y = radius; y < image.height() - radius; ++y)
    {
      for (auto x = radius; x < image.width() - radius; ++x)
      {
        auto value = image(x, y);
        for (auto v = y - radius; v <= y + radius; ++v)
          for (auto u = x - radius; u <= x + radius; ++u)
            value = std::min(value, image(u, v));
        image_eroded(x, y) = value;
      }
    }

    return image_eroded;
  }

  inline auto dilate(const Image<std::uint8_t>& image, int radius)
  {
    auto image_dilated = Image<std::uint8_t>{image.sizes()};

    for (auto y = radius; y < image.height() - radius; ++y)
    {
      for (auto x = radius; x < image.width() - radius; ++x)
      {
        auto value = image(x, y);
        for (auto v = y - radius; v <= y + radius; ++v)
          for (auto u = x - radius; u <= x + radius; ++u)
            value = std::max(value, image(u, v));
        image_dilated(x, y) = value;
      }
    }

    return image_dilated;
  }

}  // namespace DO::Sara
