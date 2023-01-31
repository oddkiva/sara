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

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Kalpana/EasyGL.hpp>

#include <string>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

  struct PointCloudRenderer
  {
    //! @brief Shader data structure are understood as optional values.
    //! @{
    Shader _vertex_shader;
    Shader _fragment_shader;
    //! @}

    //! The shader program is understood as an optional value.
    ShaderProgram _shader_program;

    auto initialize(const std::string& vertex_shader_source,
                    const std::string& fragment_shader_source) -> void
    {
      _vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);
      _fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                          fragment_shader_source);

      _shader_program.create();
      _shader_program.attach(_vertex_shader, _fragment_shader);

#ifndef __EMSCRIPTEN__
      _shader_program.use();
      _shader_program.detach();
      _vertex_shader.destroy();
      _fragment_shader.destroy();
#endif
    }

    auto destroy_gl_objects()
    {
      _shader_program.detach();
      _shader_program.clear();
    }
  };

  //! @}

}  // namespace DO::Kalpana::GL
