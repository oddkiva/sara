#pragma once

#include <drafts/OpenCL/GL/PixelBuffer.hpp>

#ifdef __APPLE__
# include <OpenGL/gl3.h>
#else
# include <gl/glew.h>
#endif


namespace DO::Sara { namespace GL {

  template <typename T>
  struct PixelTraits;


  template <>
  struct PixelTraits<float>
  {
    enum
    {
      ChannelType = GL_FLOAT,
      PixelFormat = GL_INTENSITY32F_ARB,
      ColorSpace = GL_INTENSITY
    };
  };

  template <>
  struct PixelTraits<Rgb8>
  {
    enum
    {
      ChannelType = GL_UNSIGNED_BYTE,
      PixelFormat = GL_RGB,
      ColorSpace = GL_INTENSITY
    };
  };

  struct Texture2D
  {
    void generate()
    {
      if (!object)
        glGenTextures(1, &object);
    }

    void destroy()
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

    void bind() const
    {
      glBindTexture(GL_TEXTURE_2D, object);
    }

    void unbind() const
    {
      glBindTexture(GL_TEXTURE_2D, 0);
    }

    void set_border_type(GLenum border_type = GL_REPEAT)
    {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, border_type);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, border_type);
    }

    void set_interpolation_type(GLenum type = GL_LINEAR)
    {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, type);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, type);
    }

    template <typename T>
    void initialize_data(const Image<T>& image, int mipmap_level = 0)
    {
      glTexImage2D(GL_TEXTURE_2D, mipmap_level, GL::PixelTraits<T>::PixelFormat,
                   image.width(), image.height(),
                   /* border */ 0, GL::PixelTraits<T>::PixelFormat,
                   GL::PixelTraits<T>::ChannelType,
                   reinterpret_cast<const void*>(image.data()));
    }

    template <typename T>
    void setup_with_pretty_defaults(const Image<T>& image, int mipmap_level = 0)
    {
      bind();
      set_border_type(GL_REPEAT);
      set_interpolation_type(GL_LINEAR);
      initialize_data(image, mipmap_level);
    }


  private:
    GLuint object{0};
  };

} /* namespace GL */
} /* namespace DO::Sara */
