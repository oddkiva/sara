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

#include "MetricGridRenderer.hpp"
#include "Scene.hpp"


std::unique_ptr<MetricGridRenderer> MetricGridRenderer::_renderer = nullptr;


auto MetricGridRenderer::instance() -> MetricGridRenderer&
{
  if (_renderer == nullptr)
    _renderer.reset(new MetricGridRenderer{});
  return *_renderer;
}

auto MetricGridRenderer::add_line_segment(const Eigen::Vector2f& a,
                                          const Eigen::Vector2f& b,
                                          float thickness,
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

auto MetricGridRenderer::initialize() -> void
{
  // Create a vertex shader.
  const std::map<std::string, int> arg_pos = {{"in_coords", 0}};

  const auto vertex_shader_source = R"shader(#version 300 es
    // The metric grid are in the vehicle coordinate frame.
    layout (location = 0) in vec2 in_coords;

    // Intrinsic camera parameters
    uniform vec2 image_sizes;
    uniform mat3 K;
    uniform vec2 k;
    uniform vec2 p;
    uniform float xi;

    // Extrinsic camera parameters
    uniform mat4 C;  // as in C = [R, t]

    // Scene projection.
    uniform mat4 view;
    uniform mat4 projection;

    vec2 lens_distortion(vec2 m)
    {
      // Radial component (additive).
      float r2 = dot(m, m);
      float r4 = r2 * r2;
      vec2 radial_factor = (k[0] * r2 + k[1] * r4) * m;

      // Tangential component (additive).
      float tx = 2. * p[0] * m.x * m.y + p[1] * (r2 + 2. * p[0] * m.x);
      float ty = p[0] * (r2 + 2. * p[0] * m.y) + 2. * p[1] * m.x * m.y;

      // Apply the distortion.
      vec2 delta = radial_factor + vec2(tx, ty);

      return delta;
    }

    vec3 project_to_image(vec2 coords)
    {
      // Apply the camera extrinsic parameters.
      vec4 Xc = C * vec4(coords.x, coords.y, 0., 1.);

      // 3D ray in the camera frame.
      vec3 Xs = normalize(Xc.xyz);

      // Change coordinates.
      vec3 Xe = Xs + vec3(0., 0., xi);

      // Project to the camera film plane.
      Xe /= Xe.z;
      vec2 m = Xe.xy;

      // Add distortion.
      vec2 m_distorted = m + lens_distortion(m);

      vec3 p = K * vec3(m_distorted, 1.);

      return p;
    }

    vec2 to_texture_coordinates(vec2 pn)
    {
      pn /= image_sizes.y;
      float aspect_ratio = image_sizes.x / image_sizes.y;
      pn.x -= 0.5 * aspect_ratio;
      pn.y = 0.5 - pn.y;
      return pn;
    }

    void main()
    {
      vec3 p = project_to_image(in_coords);
      vec2 pn = (p / p.z).xy;
      pn = to_texture_coordinates(pn);

      gl_Position = projection * view * vec4(pn, 0.0, 1.0);
      // Modify the depth.
      gl_Position.z = -0.15;
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
      frag_color = vec4(0.3, 0.0, 0.8, 0.4);
    }
    )shader";
  _fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                      fragment_shader_source);

  _shader_program.create();
  _shader_program.attach(_vertex_shader, _fragment_shader);

  // Allocate the GL buffers.
  _vao.generate();
  _vbo.generate();
  _ebo.generate();
}

auto MetricGridRenderer::transfer_line_tesselation_to_gl_buffers() -> void
{
  glBindVertexArray(_vao);

  // Copy vertex coordinates and triangles to the GPU.
  _vbo.bind_vertex_data(_vertices);
  _ebo.bind_triangles_data(_triangles);

  // Map the vertex coordinates to the argument position for the vertex shader.
  static constexpr auto row_bytes = 2 * sizeof(float);
  glVertexAttribPointer(0, 2 /* 2D points */, GL_FLOAT, GL_FALSE, row_bytes, 0);
  glEnableVertexAttribArray(0);
}

auto MetricGridRenderer::destroy_gl_objects() -> void
{
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();

  _vao.destroy();
  _vbo.destroy();
  _ebo.destroy();
}

auto MetricGridRenderer::render() -> void
{
  _shader_program.use(true);

  // Set the projection-model-view matrix uniforms.
  {
    auto& scene = Scene::instance();

    _shader_program.set_uniform_texture("image", scene._texture);
    _shader_program.set_uniform_vector2f("image_sizes", scene._image_sizes.data());

    _shader_program.set_uniform_matrix4f("view", scene._model_view.data());
    _shader_program.set_uniform_matrix4f("projection",
                                         scene._projection.data());

    _shader_program.set_uniform_matrix4f("C", _extrinsics.data());
    _shader_program.set_uniform_matrix3f("K", _intrinsics.K.data());
    _shader_program.set_uniform_vector2f("k", _intrinsics.radial_distortion_coefficients.data());
    _shader_program.set_uniform_vector2f("p", _intrinsics.tangential_distortion_coefficients.data());
    _shader_program.set_uniform_param("xi", _intrinsics.xi);

  }

  glBindVertexArray(_vao);
  glDrawElements(GL_TRIANGLES, _triangles.size(), GL_UNSIGNED_INT, 0);
  glDrawElements(GL_LINES, _triangles.size(), GL_UNSIGNED_INT, 0);
}
