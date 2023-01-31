// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Kalpana/EasyGL/Texture.hpp>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

  struct ImageTexture
  {
    Texture2D _texture;
    GLuint _texture_unit;

    auto set_texture(const Sara::ImageView<DO::Sara::Rgb8>& image,
                     GLuint texture_unit) -> void;

    auto destroy() -> void;
  };

  //! @}

}  // namespace DO::Kalpana::GL


