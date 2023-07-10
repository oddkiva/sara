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

  class TexturedImage2D
  {
  public:
    TexturedImage2D() = default;

    auto initialize(const Sara::ImageView<Sara::Rgb8>& image,
                    const GLuint texture_unit) -> void;

    auto destroy() -> void;

    auto reset(const Sara::ImageView<Sara::Rgb8>& image,
               const int mipmap_level = 0) -> void
    {
      _texture.bind();
      _texture.initialize_data(image, mipmap_level);
    }

    //! @brief Texture object.
    auto texture_object() -> Texture2D&
    {
      return _texture;
    }

    auto texture_object() const -> const Texture2D&
    {
      return _texture;
    }

    //! @brief Texture unit index for shader operations.
    auto texture_unit() const -> GLuint
    {
      return _texture_unit;
    }

  private:
    //! @brief Wrapped OpenGL texture object.
    Texture2D _texture;
    //! @brief Texture unit for shader operations.
    GLuint _texture_unit;
  };

  //! @}

}  // namespace DO::Kalpana::GL
