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

#pragma once

#include <DO/Sara/Core/Image.hpp>


namespace DO::Sara {

  auto read_heif_file_as_interleaved_rgb_image(const std::string& filepath)
      -> Image<Rgb8>;

  auto write_heif_file(const ImageView<Rgb8>&, const std::string&,
                       const int quality) -> void;

}  // namespace DO::Sara
