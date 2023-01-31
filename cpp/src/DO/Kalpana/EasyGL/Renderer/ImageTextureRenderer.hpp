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

#include <DO/Kalpana/EasyGL/Shader.hpp>
#include <DO/Kalpana/EasyGL/Objects/ImageTexture.hpp>
#include <DO/Kalpana/EasyGL/Objects/Quad.hpp>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

  struct ImageTextureRenderer
  {
    // OpenGL shaders.
    Shader _vertex_shader;
    Shader _fragment_shader;
    ShaderProgram _shader_program;

    auto initialize() -> void;
    auto destroy_gl_objects() -> void;

    auto render(const ImageTexture& what, const Quad& where) -> void;
  };

  //! @}

}  // namespace DO::Kalpana::GL
