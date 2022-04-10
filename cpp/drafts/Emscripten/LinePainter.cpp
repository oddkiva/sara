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

#include "LinePainter.hpp"
#include "Scene.hpp"


std::unique_ptr<LinePainter> LinePainter::_painter = nullptr;


auto LinePainter::instance() -> LinePainter&
{
  if (_painter == nullptr)
    _painter.reset(new LinePainter{});
  return *_painter;
}

auto LinePainter::add_line_segment(const Eigen::Vector2f& a,
                                   const Eigen::Vector2f& b, float thickness,
                                   float antialias_radius) -> void
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

  const auto n = _vertices.size() / 2;

  // Store the quad vertices.
  _vertices.push_back(a0.x());
  _vertices.push_back(a0.y());

  _vertices.push_back(a1.x());
  _vertices.push_back(a1.y());

  _vertices.push_back(b0.x());
  _vertices.push_back(b0.y());

  _vertices.push_back(b1.x());
  _vertices.push_back(b1.y());

  // Tesselate the quad into two right triangles.
  _triangles.push_back(n + 0);  // a0
  _triangles.push_back(n + 1);  // b0
  _triangles.push_back(n + 2);  // a1

  _triangles.push_back(n + 2);  // b0
  _triangles.push_back(n + 3);  // b1
  _triangles.push_back(n + 1);  // a1
}

auto LinePainter::add_line_segment_in_pixel_coordinates(
    const Eigen::Vector2f& a1, const Eigen::Vector2f& b1,  //
    float thickness, float antialias_radius) -> void
{
  const auto& scene = Scene::instance();
  const auto& image_sizes = scene._image_sizes;
  const auto h = static_cast<float>(image_sizes.y());

  Eigen::Vector2f a = a1 / h;
  Eigen::Vector2f b = b1 / h;

  a.y() = 1 - a.y();
  b.y() = 1 - b.y();

  add_line_segment(a, b, thickness / h, antialias_radius / h);
}


auto LinePainter::initialize() -> void
{
  // Create a vertex shader.
  const std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                              {"in_color", 1},   //
                                              {"out_color", 0}};

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
      gl_Position = projection * view * vec4(to_texture_coordinates(in_coords), 0., 1.);
    }
    )shader";
  _vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);

  // Create a fragment shader.
  const auto fragment_shader_source = R"shader(#version 300 es
    #ifdef GL_ES
    precision highp float;
    #endif

    out vec4 frag_color;

    void main()
    {
      frag_color = vec4(0.8, 0.4, 0.4, 0.8);
    }
    )shader";
  _fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                      fragment_shader_source);

  _shader_program.create();
  _shader_program.attach(_vertex_shader, _fragment_shader);
  _shader_program.validate();

  // Clearing the shaders after attaching them to the shader program does not
  // work on WebGL 2.0/OpenGL ES 3.0... I don't know why.
  //
  // _shader_program.use();
  // _shader_program.detach();
  // _vertex_shader.destroy();
  // _fragment_shader.destroy();

  // Allocate the GL buffers.
  _vao.generate();
  _vbo.generate();
  _ebo.generate();
}

auto LinePainter::transfer_line_tesselation_to_gl_buffers() -> void
{
  glBindVertexArray(_vao);

  // Copy vertex coordinates and triangles to the GPU.
  _vbo.bind_vertex_data(_vertices);
  _ebo.bind_triangles_data(_triangles);

  // Map the parameters to the argument position for the vertex shader.
  //
  // Vertex coordinates.
  static constexpr auto row_bytes = 2 * sizeof(float);
  glVertexAttribPointer(0, 2 /* 2D points */, GL_FLOAT, GL_FALSE, row_bytes, 0);
  glEnableVertexAttribArray(0);
}

auto LinePainter::destroy_gl_objects() -> void
{
  _shader_program.use();
  _shader_program.detach();
  _shader_program.clear();
  _vertex_shader.destroy();
  _fragment_shader.destroy();

  _vao.destroy();
  _vbo.destroy();
  _ebo.destroy();
}

auto LinePainter::render() -> void
{
  _shader_program.use(true);

  // Set the projection-model-view matrix uniforms.
  auto& scene = Scene::instance();
  _shader_program.set_uniform_vector2f("image_sizes",
                                       scene._image_sizes.data());
  _shader_program.set_uniform_matrix4f("view", scene._model_view.data());
  _shader_program.set_uniform_matrix4f("projection", scene._projection.data());

  glBindVertexArray(_vao);
  glDrawElements(GL_TRIANGLES, _triangles.size(), GL_UNSIGNED_INT, 0);
}
