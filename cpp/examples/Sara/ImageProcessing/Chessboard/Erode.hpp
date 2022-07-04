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

#include <DO/Sara/Core/Image.hpp>


namespace DO::Sara {

  inline auto binary_erode_3x3(const ImageView<std::uint8_t>& src,
                               ImageView<std::uint8_t>& dst) -> void
  {
#pragma omp parallel for collapse(2)
    for (auto y = 1; y < src.height() - 1; ++y)
    {
      for (auto x = 1; x < src.width() - 1; ++x)
      {
        auto val = src(x, y) == 0;
        for (auto v = -1; v <= 1; ++v)
          for (auto u = -1; u <= 1; ++u)
            val = val && src(x + u, y + v) == 0;
        dst(x, y) = val ? 0 : 255;
      }
    }
  }

}  // namespace DO::Sara
