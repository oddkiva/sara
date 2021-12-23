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

#include <DO/Sara/Core/Image.hpp>


namespace DO::Sara {

  auto from_rgb8_to_gray32f(const ImageView<Rgb8>& src, ImageView<float>& dst)
      -> void;

  inline auto from_rgb8_to_gray32f(const ImageView<Rgb8>& src)
  {
    auto dst = Image<float>{src.sizes()};
    from_rgb8_to_gray32f(src, dst);
    return dst;
  }

}  // namespace DO::Sara
