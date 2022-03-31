// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <drafts/OpenCL/GL/OpenGL.hpp>
#include <drafts/OpenCL/GL/PixelBuffer.hpp>

#include <DO/Sara/Core/Pixel.hpp>


namespace DO::Sara { namespace GL {

  //! @addtogroup OpenGL
  //! @{

  template <typename T>
  struct PixelTraits;


  template <>
  struct PixelTraits<float>
  {
    enum
    {
      ChannelType = GL_FLOAT,
      PixelFormat = GL_DEPTH_COMPONENT
    };
  };

  template <>
  struct PixelTraits<Rgb8>
  {
    enum
    {
      ChannelType = GL_UNSIGNED_BYTE,
      PixelFormat = GL_RGB
    };
  };

  struct Texture2D
  {
    inline void generate()
    {
      if (!object)
        glGenTextures(1, &object);
    }

    inline void destroy()
    {
      if (object)
      {
        glBindTexture(1, object);
        glDeleteTextures(1, &object);
      }
    }

    inline operator GLuint() const
    {
      return object;
    }

    inline void bind() const
    {
      glBindTexture(GL_TEXTURE_2D, object);
    }

    inline void unbind() const
    {
      glBindTexture(GL_TEXTURE_2D, 0);
    }

    inline void set_border_type(GLenum border_type = GL_REPEAT)
    {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, border_type);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, border_type);
    }

    inline void set_interpolation_type(GLenum type = GL_LINEAR)
    {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, type);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, type);
    }

    template <typename T>
    inline void initialize_data(const ImageView<T>& image, int mipmap_level = 0)
    {
      glTexImage2D(GL_TEXTURE_2D, mipmap_level, GL::PixelTraits<T>::PixelFormat,
                   image.width(), image.height(),
                   /* border */ 0, GL::PixelTraits<T>::PixelFormat,
                   GL::PixelTraits<T>::ChannelType,
                   reinterpret_cast<const void*>(image.data()));
      glGenerateMipmap(GL_TEXTURE_2D);
    }

    template <typename T>
    inline void setup_with_pretty_defaults(const ImageView<T>& image,
                                           int mipmap_level = 0)
    {
      generate();
      bind();
      set_border_type(GL_CLAMP_TO_EDGE);
      set_interpolation_type(GL_LINEAR);
      initialize_data(image, mipmap_level);
    }

    GLuint object{0};
  };

  //! @}

}}  // namespace DO::Sara::GL
