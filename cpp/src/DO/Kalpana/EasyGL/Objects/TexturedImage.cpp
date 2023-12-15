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

#include <DO/Kalpana/EasyGL/Objects/TexturedImage.hpp>


using namespace DO::Kalpana::GL;

namespace sara = DO::Sara;


auto TexturedImage2D::initialize(const sara::ImageView<sara::Rgb8>& image_view,
                                 const GLuint texture_unit) -> void
{
  // Bind the texture unit: GL_TEXTURE0 + i.
  _texture_unit = texture_unit;
  glActiveTexture(GL_TEXTURE0 + _texture_unit);

  // Initialize the texture object and bind it.
  //
  // The texture unit GL_TEXTURE0 + i will be associated to this texture object.
  if (!_texture)
    _texture.generate();
  _texture.bind();

  // Copy the image to the GPU texture.
  _texture.initialize_data(image_view, 0);

  // Set the image display options.
  _texture.set_border_type(GL_CLAMP_TO_EDGE);
  _texture.set_interpolation_type(GL_LINEAR);
}

auto TexturedImage2D::destroy() -> void
{
  _texture.destroy();
}
