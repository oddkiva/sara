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

#include <DO/Kalpana/EasyGL/Renderer/CheckerboardRenderer.hpp>


using namespace DO::Kalpana::GL;


CheckerboardRenderer::CheckerboardRenderer(const int rows, const int cols,
                                           const float scale,
                                           const float y_origin)
  : _rows{rows}
  , _cols{cols}
{
  _vertices = Sara::Tensor_<float, 2>{{4 * rows * cols, 6}};
  _triangles = Sara::Tensor_<unsigned int, 2>{{2 * rows * cols, 3}};

  auto v_mat = _vertices.matrix();
  auto t_mat = _triangles.matrix();
  for (auto i = 0; i < _rows; ++i)
  {
    for (auto j = 0; j < cols; ++j)
    {
      const auto ij = cols * i + j;

      // Coordinates.
      //
      // clang-format off
      v_mat.block(4 * ij, 0, 4, 3) <<  // coords
          i +  0.5f, y_origin, j + 0.5f,    // top-right
          i +  0.5f, y_origin, j + -0.5f,   // bottom-right
          i + -0.5f, y_origin, j + -0.5f,  // bottom-left
          i + -0.5f, y_origin, j + 0.5f;   // top-left
      // clang-format on

      // Set colors.
      if (i % 2 == 0 && j % 2 == 0)
        v_mat.block(4 * ij, 3, 4, 3).setZero();
      else if (i % 2 == 0 && j % 2 == 1)
        v_mat.block(4 * ij, 3, 4, 3).setOnes();
      else if (i % 2 == 1 && j % 2 == 0)
        v_mat.block(4 * ij, 3, 4, 3).setOnes();
      else  // (i % 2 == 1 and j % 2 == 0)
        v_mat.block(4 * ij, 3, 4, 3).setZero();

      // vertex indices for each triangle that forms the quad
      //
      // clang-format off
      t_mat.block(2 * ij, 0, 2, 3) <<
        4 * ij + 0, 4 * ij + 1, 4 * ij + 2,
        4 * ij + 2, 4 * ij + 3, 4 * ij + 0;
      // clang-format on
    }
  }
  // Translate.
  v_mat.col(0).array() -= _rows / 2.f;
  v_mat.col(2).array() -= _cols / 2.f;
  // Rescale.
  v_mat.leftCols(3) *= scale;
}

auto CheckerboardRenderer::initialize_geometry() -> void
{
  const auto row_bytes = [](const Sara::TensorView_<float, 2>& data) {
    return static_cast<GLsizei>(data.size(1) * sizeof(float));
  };
  const auto float_pointer = [](int offset) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  _vao.generate();
  _vbo.generate();
  _ebo.generate();

  // Specify the vertex attributes here.
  glBindVertexArray(_vao);

  // Copy the vertex data into the GPU buffer object.
  _vbo.bind_vertex_data(_vertices);

  // Copy the triangles data into the GPU buffer object.
  _ebo.bind_triangles_data(_triangles);

  // Specify that the vertex shader param 0 corresponds to the first 3
  // float data of the buffer object.
  glVertexAttribPointer(0, 3 /* 3D points */, GL_FLOAT, GL_FALSE,
                        row_bytes(_vertices), float_pointer(0));
  glEnableVertexAttribArray(0);

  // Specify that the vertex shader param 1 corresponds to the first 3
  // float data of the buffer object.
  glVertexAttribPointer(1, 3 /* 3D colors */, GL_FLOAT, GL_FALSE,
                        row_bytes(_vertices), float_pointer(3));
  glEnableVertexAttribArray(1);

  // Unbind the vbo to protect its data.
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

auto CheckerboardRenderer::initialize_shader_program() -> void
{
  const auto vertex_shader_source = R"shader(#version 330 core
    layout (location = 0) in vec3 in_coords;
    layout (location = 1) in vec3 in_color;

    uniform mat4 transform;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 out_color;

    void main()
    {
      gl_Position = projection * view * transform * vec4(in_coords, 1.0);
      gl_PointSize = 200.0;
      out_color = in_color;
    }
  )shader";
  _vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);


  const auto fragment_shader_source = R"shader(#version 330 core
    in vec3 out_color;
    out vec4 frag_color;

    void main()
    {
      frag_color = vec4(out_color, 0.1);
    }
  )shader";
  _fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                      fragment_shader_source);

  _shader_program.create();
  _shader_program.attach(_vertex_shader, _fragment_shader);

  _transform_loc = _shader_program.get_uniform_location("transform");
  _view_loc = _shader_program.get_uniform_location("view");
  _projection_loc = _shader_program.get_uniform_location("projection");

#if !defined(__EMSCRIPTEN__)
  _shader_program.use();
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();
#endif
}

auto CheckerboardRenderer::initialize() -> void
{
  initialize_geometry();
  initialize_shader_program();
}

auto CheckerboardRenderer::destroy() -> void
{
  _shader_program.use();
  _shader_program.detach();
#if defined(__EMSCRIPTEN__)
  _vertex_shader.destroy();
  _fragment_shader.destroy();
#endif
  _shader_program.clear();

  _vao.destroy();
  _vbo.destroy();
  _ebo.destroy();
}

auto CheckerboardRenderer::render(const Eigen::Matrix4f& transform,
                                  const Eigen::Matrix4f& model_view,
                                  const Eigen::Matrix4f& projection) -> void
{
  _shader_program.use();

  // Pass the parameters to the shader program.
  _shader_program.set_uniform_matrix4f(_transform_loc, transform.data());
  _shader_program.set_uniform_matrix4f(_view_loc, model_view.data());
  _shader_program.set_uniform_matrix4f(_projection_loc, projection.data());

  glBindVertexArray(_vao);
  glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(_triangles.size()),
                 GL_UNSIGNED_INT, 0);
}
