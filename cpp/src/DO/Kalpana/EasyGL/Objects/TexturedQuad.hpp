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

#pragma once

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Kalpana/EasyGL/Buffer.hpp>
#include <DO/Kalpana/EasyGL/VertexArray.hpp>

#include <array>
#include <cstdint>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

  //! @brief Textured Quad
  //!
  //! We choose to make a simple implementation where we do bookkeep the vertex
  //! data on the host side (CPU).
  class TexturedQuad
  {
  public:
    //! @brief The vertex dimension specified in plain English.
    //! @{
    static constexpr auto loc_coords_dim = 3;
    static constexpr auto tex_coords_dim = 2;
    static constexpr auto vertex_dim = loc_coords_dim + tex_coords_dim;
    //! @}

    //! @brief Number of vertices and triangles specified in plain English.
    //! @{
    static constexpr auto num_vertices = 4;
    static constexpr auto num_triangles = 2;
    //! @}

    //! @brief By default, initialize a canonical unit square centered at the
    //! screen.
    TexturedQuad();

    //! @brief Default destructor.
    ~TexturedQuad() = default;

    //! @brief Initialize OpenGL resources (VAO, VBO and EBO).
    //!
    //! N.B.: GLEW must be initialized before!
    auto initialize() -> void;

    //! @brief Destroy OpenGL resources.
    //!
    //! N.B.: GLEW must be initialized before!
    auto destroy() -> void;

    //! @brief Bind the vertex array object.
    //!
    //! You need to call this if you need to update the quad vertex data on
    //! OpenGL side.
    auto bind_vertex_array() -> void
    {
      glBindVertexArray(_vao);
    }

    //! @brief Reupload the vertex data from host to OpenGL.
    auto upload_vertex_data_from_host_to_gl() -> void
    {
      _vbo.bind_vertex_data(host_vertices());
    }

    auto host_vertices() -> Sara::TensorView_<float, 2>
    {
      return Sara::TensorView_<float, 2>{_host_vertex_data.data(),
                                         {num_vertices, vertex_dim}};
    }

    auto host_triangles() -> Sara::TensorView_<std::uint32_t, 2>
    {
      return Sara::TensorView_<std::uint32_t, 2>{_host_triangle_data.data(),
                                                 {num_triangles, 3}};
    }

    auto host_triangles() const -> Sara::TensorView_<const std::uint32_t, 2>
    {
      return Sara::TensorView_<const std::uint32_t, 2>{
          _host_triangle_data.data(), {num_triangles, 3}};
    }

    auto vertex_row_byte_size() const -> GLsizei
    {
      return static_cast<GLsizei>(vertex_dim * sizeof(float));
    };

    auto vao() const -> const VertexArray&
    {
      return _vao;
    }

  private:
    //! @brief Host vertex data.
    std::array<float, num_vertices * vertex_dim> _host_vertex_data;
    //! @brief Host triangle data.
    std::array<std::uint32_t, num_triangles * 3 /* vertices per triangles */>
        _host_triangle_data;

    //! @brief OpenGL GPU device data.
    //!
    //! @brief Vertex data layout specified for OpenGL.
    VertexArray _vao;
    //! @brief Vertex data.
    Buffer _vbo;
    //! @brief Triangle data.
    Buffer _ebo;
  };

  //! @}

}  // namespace DO::Kalpana::GL
