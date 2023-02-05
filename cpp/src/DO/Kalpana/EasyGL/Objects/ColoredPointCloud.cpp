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

#include <DO/Kalpana/EasyGL/Objects/ColoredPointCloud.hpp>


using namespace DO::Kalpana::GL;


auto ColoredPointCloud::initialize() -> void
{
  // Reminder: the VAO describes the geometry data layout and maps the data
  // to the shader uniform variables.
  _vao.generate();
  // Reminder: the VBO is a buffer containing the geometry data.
  _vbo.generate();
}

auto ColoredPointCloud::destroy() -> void
{
  _vao.destroy();
  _vbo.destroy();
}

auto ColoredPointCloud::upload_host_data_to_gl(
    const Sara::TensorView_<float, 2>& point_cloud) -> void
{
  static const auto vertex_data_byte_size = [&point_cloud]() {
    return static_cast<GLsizei>(point_cloud.size(1) * sizeof(float));
  };
  static const auto float_pointer = [](int offset = 0) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  // Important: specify the vertex attribute object that defines the point
  // cloud data.
  glBindVertexArray(_vao);

  // Copy the point cloud data from the host to OpenGL.
  _vbo.bind_vertex_data(point_cloud);

  // Vertex coordinates.
  glVertexAttribPointer(coords_attr_index, coords_dim,  //
                        GL_FLOAT, GL_FALSE,             //
                        vertex_data_byte_size(), float_pointer(0));
  glEnableVertexAttribArray(coords_attr_index);

  // Color values.
  glVertexAttribPointer(color_attr_index, color_dim,  //
                        GL_FLOAT, GL_FALSE,           //
                        vertex_data_byte_size(), float_pointer(coords_dim));
  glEnableVertexAttribArray(color_attr_index);
}
