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
#include <DO/Kalpana/EasyGL/Objects/TexturedQuad.hpp>
#include <DO/Kalpana/EasyGL/Shader.hpp>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

  struct TextureRenderer
  {
    //! @brief OpenGL shaders.
    //! @{
    Shader _vertex_shader;
    Shader _fragment_shader;
    ShaderProgram _shader_program;
    //! @}

    //! brief Uniform locations.
    //! @{
    GLint _model_view_loc;
    GLint _projection_loc;
    GLint _image_loc;
    //! @}

    auto initialize() -> void;

    auto destroy() -> void;

    auto render(TexturedImage2D& what, TexturedQuad& where,
                const Eigen::Matrix4f& model_view,
                const Eigen::Matrix4f& projection) -> void;
  };

  //! @}

}  // namespace DO::Kalpana::GL
