// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Kalpana/EasyGL/OpenGL.hpp>
#include <DO/Kalpana/EasyGL/Texture.hpp>


namespace DO::Kalpana::GL {

  //! @addtogroup OpenGL
  //! @{

  struct Renderbuffer
  {
    inline void generate()
    {
      if (!object)
        glGenRenderbuffers(1, &object);
    }

    inline auto destroy() -> void
    {
      if (object)
        glDeleteRenderbuffers(1, &object);
    }

    inline operator GLuint() const
    {
      return object;
    }

    inline auto bind() const -> void
    {
      glBindRenderbuffer(GL_RENDERBUFFER, object);
    }

    inline auto unbind() const -> void
    {
      glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }

    inline auto set(int width, int height, GLuint type) -> void
    {
      glRenderbufferStorage(GL_RENDERBUFFER, /* GL_DEPTH24_STENCIL8 */ type,
                            width, height);
    }

    GLuint object{0};
  };

  //! @}

}  // namespace DO::Kalpana::GL
