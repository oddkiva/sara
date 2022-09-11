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

#include <DO/Kalpana/EasyGL/OpenGL.hpp>

#include <array>
#include <numeric>


namespace DO::Kalpana::GL {

  //! @addtogroup OpenGL
  //! @{

  template <typename T, int N = 2>
  class PixelBuffer
  {
  public:
    typedef T pixel_type;
    enum
    {
      Dimension = N
    };

  public:
    PixelBuffer(size_t width, size_t height, const T* data = nullptr)
      : _dims({width, height})
    {
      glGenBuffers(1, &_pbo_id);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo_id);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(T), data,
                   GL_DYNAMIC_DRAW);
    }

    PixelBuffer(size_t width, size_t height, size_t depth,
                const T* data = nullptr)
      : _dims({width, height, depth})
    {
      glGenBuffers(1, &_pbo_id);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo_id);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * depth * sizeof(T),
                   data, GL_DYNAMIC_DRAW);
    }

    ~PixelBuffer()
    {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glDeleteBuffers(1, &_pbo_id);
    }

    inline operator GLuint() const
    {
      return _pbo_id;
    }

    inline size_t width() const
    {
      return _dims[0];
    }

    inline size_t height() const
    {
      return _dims[1];
    }

    inline size_t depth() const
    {
      return _dims[2];
    }

    inline size_t byte_size() const
    {
      return sizeof(pixel_type) * std::accumulate(_dims.begin(), _dims.end(),
                                                  static_cast<size_t>(1),
                                                  std::multiplies<size_t>());
    }

    inline void bind() const
    {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo_id);
    }

    inline void unbind() const
    {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    void unpack(const T* data) const
    {
      glBufferData(GL_PIXEL_UNPACK_BUFFER, byte_size(), data, GL_DYNAMIC_DRAW);
    }

  private:
    GLuint _pbo_id;
    std::array<size_t, N> _dims;
  };

  //! @}

}  // namespace DO::Kalpana::GL
