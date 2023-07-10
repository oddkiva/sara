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

#include "LineRenderer.hpp"


auto LineRenderer::LineHostData::add_line_segment(const Eigen::Vector2f& a,
                                                  const Eigen::Vector2f& b,
                                                  float thickness,
                                                  float antialias_radius)
    -> void
{
  // First calculate the quad vertices.
  const Eigen::Vector2f t = (b - a).normalized();
  const Eigen::Vector2f o = Eigen::Vector2f(-t.y(), t.x());

  const auto& w = thickness;
  const auto& r = antialias_radius;
  const Eigen::Vector2f a0 = a - (w * 0.5f + r) * (t - o);
  const Eigen::Vector2f a1 = a - (w * 0.5f + r) * (t + o);
  const Eigen::Vector2f b0 = b + (w * 0.5f + r) * (t + o);
  const Eigen::Vector2f b1 = b + (w * 0.5f + r) * (t - o);

  const auto n = static_cast<std::uint32_t>(_vertices.size() / 2);

  // Store the quad vertices.
  _vertices.push_back(a0.x());
  _vertices.push_back(a0.y());
  std::cout << a0.transpose() << std::endl;

  _vertices.push_back(a1.x());
  _vertices.push_back(a1.y());
  std::cout << a1.transpose() << std::endl;

  _vertices.push_back(b0.x());
  _vertices.push_back(b0.y());
  std::cout << b0.transpose() << std::endl;

  _vertices.push_back(b1.x());
  _vertices.push_back(b1.y());
  std::cout << b1.transpose() << std::endl;

  // Tesselate the quad into two right triangles.
  _triangles.push_back(n + 0);  // a0
  _triangles.push_back(n + 1);  // b0
  _triangles.push_back(n + 2);  // a1

  _triangles.push_back(n + 2);  // b0
  _triangles.push_back(n + 3);  // b1
  _triangles.push_back(n + 1);  // a1

  for (const auto& i : _triangles)
    std::cout << i << " ";
  std::cout << std::endl;
}

auto LineRenderer::LineHostData::add_line_segment_in_pixel_coordinates(
    const Eigen::Vector2f& a1,           //
    const Eigen::Vector2f& b1,           //
    const Eigen::Vector2f& image_sizes,  //
    float thickness,                     //
    float antialias_radius) -> void
{
  const auto h = static_cast<float>(image_sizes.y());

  Eigen::Vector2f a = a1 / h;
  Eigen::Vector2f b = b1 / h;

  a.y() = 1 - a.y();
  b.y() = 1 - b.y();

  add_line_segment(a, b, thickness / h, antialias_radius / h);
}


auto LineRenderer::LineShaderData::set_data(const LineHostData& host_data)
    -> void
{
  // Bind the VAO.
  if (_vao == 0)
    _vao.generate();
  // Allocate the GL buffers.
  if (_vbo == 0)
  {
    std::cout << "Generating VBO" << std::endl;
    _vbo.generate();
  }
  if (_ebo == 0)
    _ebo.generate();

  _triangle_index_count = host_data._triangles.size();
  std::cout << _triangle_index_count << std::endl;

  glBindVertexArray(_vao);
  glEnableVertexAttribArray(0);

  // Copy vertex coordinates and triangles to the GPU.
  _vbo.bind_vertex_data(host_data._vertices);
  _ebo.bind_triangles_data(host_data._triangles);

  // Specify the VAO descriptor.
  static constexpr auto row_bytes = 2 * sizeof(float);
  glVertexAttribPointer(0, 2 /* 2D points */, GL_FLOAT, GL_FALSE, row_bytes, 0);
}

auto LineRenderer::LineShaderData::destroy() -> void
{
  _vbo.destroy();
  _ebo.destroy();
}


auto LineRenderer::initialize() -> void
{
  // Create a vertex shader.
  static const std::map<std::string, int> arg_pos = {{"in_coords", 0}};

  const auto vertex_shader_source = R"shader(#version 300 es
    layout (location = 0) in vec2 in_coords;

    uniform vec2 image_sizes;
    uniform mat4 view;
    uniform mat4 projection;

    vec2 to_texture_coordinates(vec2 pn)
    {
      float aspect_ratio = image_sizes.x / image_sizes.y;
      pn.x -= 0.5 * aspect_ratio;
      pn.y -= 0.5;
      return pn;
    }

    void main()
    {
      vec4 coords = vec4(to_texture_coordinates(in_coords), 0., 1.);
      gl_Position = projection * view * coords;
    }
  )shader";
  _vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);

  // Create a fragment shader.
  const auto fragment_shader_source = R"shader(#version 300 es
    #ifdef GL_ES
    precision highp float;
    #endif

    uniform vec4 color;

    out vec4 frag_color;

    void main()
    {
      frag_color = color;
    }
  )shader";
  _fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                      fragment_shader_source);

  _shader_program.create();
  _shader_program.attach(_vertex_shader, _fragment_shader);
  _shader_program.validate();

  // Locate the uniform variables in the compiled shader.
  _image_sizes_loc = _shader_program.get_uniform_location("image_sizes");
  _color_loc = _shader_program.get_uniform_location("color");
  _view_loc = _shader_program.get_uniform_location("view");
  _projection_loc = _shader_program.get_uniform_location("projection");

#ifndef __EMSCRIPTEN__
  // Clearing the shaders after attaching them to the shader program does not
  // work on WebGL 2.0/OpenGL ES 3.0... I don't know why.
  _shader_program.use();
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();
#endif
}

auto LineRenderer::destroy_gl_objects() -> void
{
  _shader_program.use();
  _shader_program.detach();
  _shader_program.clear();
  _vertex_shader.destroy();
  _fragment_shader.destroy();
}

auto LineRenderer::render(const ImagePlaneRenderer::ImageTexture& image_plane,
                          const LineShaderData& lines) -> void
{
  // Select the shader program.
  _shader_program.use(true);
  // Set the projection-model-view matrix uniforms.
  _shader_program.set_uniform_vector2f(_image_sizes_loc,
                                       image_plane._image_sizes.data());
  _shader_program.set_uniform_vector4f(_color_loc, lines._color.data());
  _shader_program.set_uniform_matrix4f(_view_loc,
                                       image_plane._model_view.data());
  _shader_program.set_uniform_matrix4f(_projection_loc,
                                       image_plane._projection.data());

  // Select the vertex array descriptor.
  glBindVertexArray(lines._vao);
  glDrawElements(GL_TRIANGLES,
                 static_cast<GLsizei>(lines._triangle_index_count),
                 GL_UNSIGNED_INT, 0);
}
