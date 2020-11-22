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

#include <DO/Sara/Core/Tensor.hpp>

#include <drafts/OpenCL/GL/OpenGL.hpp>


namespace DO::Sara { namespace GL {

  //! @addtogroup OpenGL
  //! @{

  struct Buffer
  {
    static void generate(Buffer* begin, Buffer* end)
    {
      glGenBuffers(int(end - begin), reinterpret_cast<GLuint*>(begin));
    }

    void generate()
    {
      Buffer::generate(this, this + 1);
    }

    void destroy()
    {
      glDeleteBuffers(1, &object);
      object = 0;
    }

    operator GLuint&()
    {
      return object;
    }

    operator GLuint() const
    {
      return object;
    }

    template <typename T>
    auto bind_vertex_data(const DO::Sara::TensorView_<T, 2>& data) const
    {
      glBindBuffer(GL_ARRAY_BUFFER, object);
      glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(T), data.data(),
                   GL_STATIC_DRAW);
    }

    template <typename T>
    auto bind_triangles_data(const DO::Sara::TensorView_<T, 2>& data) const
    {
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size() * sizeof(T),
                   data.data(), GL_STATIC_DRAW);
    }

    GLuint object{0};
  };

  //! @}

} /* namespace GL */
} /* namespace DO::Sara */
