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

#include <DO/Kalpana/EasyGL.hpp>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

  struct Buffer
  {
    inline static auto generate(Buffer* begin, Buffer* end) -> void
    {
      glGenBuffers(int(end - begin), reinterpret_cast<GLuint*>(begin));
    }

    inline auto generate() -> void
    {
      Buffer::generate(this, this + 1);
    }

    inline auto destroy() -> void
    {
      glDeleteBuffers(1, &object);
      object = 0;
    }

    inline operator GLuint&()
    {
      return object;
    }

    inline operator GLuint() const
    {
      return object;
    }

    template <typename T>
    inline auto bind_vertex_data(const std::vector<T>& data) const -> void
    {
      glBindBuffer(GL_ARRAY_BUFFER, object);
      glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(T), data.data(),
                   GL_STATIC_DRAW);
    }

    template <typename T>
    inline auto bind_vertex_data(const Sara::TensorView_<T, 2>& data) const
        -> void
    {
      glBindBuffer(GL_ARRAY_BUFFER, object);
      glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(T), data.data(),
                   GL_STATIC_DRAW);
    }

    template <typename T>
    inline auto bind_triangles_data(const std::vector<T>& data) const -> void
    {
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size() * sizeof(T),
                   data.data(), GL_STATIC_DRAW);
    }

    template <typename T>
    inline auto bind_triangles_data(const Sara::TensorView_<T, 2>& data) const
        -> void
    {
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size() * sizeof(T),
                   data.data(), GL_STATIC_DRAW);
    }

    GLuint object{0};
  };

  //! @}

}  // namespace DO::Kalpana::GL
