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

//! @file

#include <DO/Kalpana/EasyGL/Objects/TexturedQuad.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>


namespace sara = DO::Sara;


using namespace DO::Kalpana::GL;


TexturedQuad::TexturedQuad()
{
  // Encode the vertex data of the square in a tensor.
  auto vertices = host_vertices();
  // clang-format off
  vertices.flat_array() <<
  // coords              texture coords
     0.5f, -0.5f, 0.0f,  1.0f, 1.0f,  // bottom-right
     0.5f,  0.5f, 0.0f,  1.0f, 0.0f,  // top-right
    -0.5f,  0.5f, 0.0f,  0.0f, 0.0f,  // top-left
    -0.5f, -0.5f, 0.0f,  0.0f, 1.0f;  // bottom-left
  // clang-format on

  auto triangles = host_triangles();
  // clang-format off
  triangles.flat_array() <<
    0, 1, 2,
    2, 3, 0;
  // clang-format on
}

auto TexturedQuad::initialize() -> void
{
  if (!_vao)
    glGenVertexArrays(1, &_vao.object);
  if (!_vbo)
    _vbo.generate();
  if (!_ebo)
    _ebo.generate();

  glBindVertexArray(_vao);

  // Copy vertex data.
  _vbo.bind_vertex_data(host_vertices());

  // Copy geometry data.
  _ebo.bind_triangles_data(host_triangles());

  // Map the parameters to the argument position for the vertex shader.
  //
  // Vertex coordinates.
  static constexpr auto float_pointer = [](int offset = 0) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  glVertexAttribPointer(loc_attr_index, loc_coords_dim,  //
                        GL_FLOAT, GL_FALSE, vertex_row_byte_size(),
                        float_pointer());
  glEnableVertexAttribArray(0);

  // Texture coordinates.
  glVertexAttribPointer(1, tex_coords_dim,  //
                        GL_FLOAT, GL_FALSE, vertex_row_byte_size(),
                        float_pointer(loc_coords_dim));
  glEnableVertexAttribArray(1);
}

auto TexturedQuad::destroy() -> void
{
  _vao.destroy();
  _vbo.destroy();
  _ebo.destroy();
}
