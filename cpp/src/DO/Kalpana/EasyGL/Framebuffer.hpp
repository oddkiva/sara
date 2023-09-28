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

#include <DO/Kalpana/EasyGL/Renderbuffer.hpp>
#include <DO/Kalpana/EasyGL/Texture.hpp>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

  struct Framebuffer
  {
    inline void generate()
    {
      if (!object)
        glGenFramebuffers(1, &object);
    }

    inline auto destroy() -> void
    {
      if (object)
      {
        glBindFramebuffer(GL_FRAMEBUFFER, object);
        glDeleteFramebuffers(1, &object);
      }
    }

    inline operator GLuint() const
    {
      return object;
    }

    inline auto bind() const -> void
    {
      glBindFramebuffer(GL_FRAMEBUFFER, object);
    }

    inline auto unbind() const -> void
    {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    inline auto attach_color_texture(const Texture& texture, GLuint id) -> void
    {
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + id,
                             GL_TEXTURE_2D, texture, 0);
    }

    inline auto attach_renderbuffer(const Renderbuffer& rbo) -> void
    {
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                GL_RENDERBUFFER, rbo);
    }

    inline auto render_to_texture(const Texture& texture, GLuint id,
                                  const Eigen::Vector2i& sizes) -> void
    {
      bind();
      attach_color_texture(texture, id);

      // // Render.
      // {
      //   glViewport(0, 0, sizes.x(), sizes.y());
      //   glClearColor(0.f, 0.f, 0.f, 1.0f);
      //   glClear(GL_COLOR_BUFFER_BIT);
      //   // draw the quad with the shader.
      //   // shader.use();
      //   // render(vao);
      // }

      unbind();
    }

    GLuint object{0};
  };

  //! @}

}  // namespace DO::Kalpana::GL
