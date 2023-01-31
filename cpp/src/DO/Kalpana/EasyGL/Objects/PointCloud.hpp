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

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Kalpana/EasyGL.hpp>

#include <map>
#include <string>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

  struct PointCloud
  {
    // Point cloud data layout specified for OpenGL.
    VertexArray _vao;
    // 3D points and other attributes (colors, normals, etc).
    Buffer _vbo;

    // Geometry memory layout.
    std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                          {"in_color", 1}};

    // Initialize the VAO and VBO.
    auto initialize_gl_objects() -> void
    {
      // Reminder: the VAO describes the geometry data layout and maps the data
      // to the shader uniform variables.
      _vao.generate();
      // Reminder: the VBO is a buffer containing the geometry data.
      _vbo.generate();
    }

    auto destroy_gl_objects() -> void
    {
      _vao.destroy();
      _vbo.destroy();
    }

    auto
    transfer_host_geometry_to_gl(const Sara::TensorView_<float, 2>& point_cloud)
        -> void
    {
      static constexpr auto point_dim = 3;
      static constexpr auto point_data_byte_size = []() {
        return static_cast<GLsizei>(point_dim * sizeof(float));
      };
      static constexpr auto float_pointer = [](int offset) {
        return reinterpret_cast<void*>(offset * sizeof(float));
      };

      // Important: specify the vertex attribute object that defines the point
      // cloud data.
      glBindVertexArray(_vao);

      // Copy the point cloud data from the host to OpenGL.
      _vbo.bind_vertex_data(point_cloud);

      // Map the parameters to the argument position for the vertex shader.
      //
      // Vertex coordinates.
      glVertexAttribPointer(arg_pos.at("in_coords"),
                            point_dim /* 3D points */,  //
                            GL_FLOAT, GL_FALSE,         //
                            point_data_byte_size(), float_pointer(0));
      glEnableVertexAttribArray(arg_pos.at("in_coords"));
    }
  };

  //! @}

}  // namespace DO::Kalpana::GL
