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
    uniform mat4 transform;
    uniform mat4 view;
    uniform mat4 projection;

    vec2 lens_distortion(vec2 m)
    {
      // Distortion.
      float k1 = k.x;
      float k2 = k.y;
      float p1 = p.x;
      float p2 = p.y;

      // Radial component (additive).
      float r2 = dot(m, m);
      float r4 = r2 * r2;
      vec2 radial_factor = (k1 * r2 + k2 * r4) * m;

      // Tangential component (additive).
      float tx = 2. * p1 * m.x * m.y + p2 * (r2 + 2. * p1 * m.x);
      float ty = p1 * (r2 + 2. * p1 * m.y) + 2. * p2 * m.x * m.y;

      // Apply the distortion.
      vec2 delta = radial_factor + vec2(tx, ty);

      return delta;
    }

    vec2 project_to_image(vec2 coords)
    {
      // 1. Project the metric grid to the image.
      //
      // 1.a) Calculate the camera coordinates.
      vec4 Xc = C * vec4(coords, 0., 1.);
      // 1.b) Project to the image.
      vec3 Xs = normalize(Xc.xyz);
      vec3 Xe = Xs + xi * vec3(0., 0., 1.);
      vec2 m = (Xe / Xe.z).xy;
      vec2 m_distorted = m + lens_distortion(m);
      vec3 p = K * vec3(m_distorted, 1.);
      vec2 pn = (p / p.z).xy;
      return pn;
    }

    vec2 rescale(vec2 pn)
    {
      float ratio = image_sizes.x / image_sizes.y;
      pn = pn - 0.5 * image_sizes;
      pn /= image_sizes.y;
      return pn;
    }

    void main()
    {
      // 2. Rescale the pixel coordinates of the metric in the normalized
      //    coordinates.
      // 3. Apply the global projection-model-view transform.
      vec2 pn = in_coords;
      gl_Position = projection * view * transform * vec4(pn, 0, 1.);
    }
    )shader";
  _vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);

  // Create a fragment shader.
  const auto fragment_shader_source = R"shader(#version 300 es
    #ifdef GL_ES
    precision highp float;
    #endif

    uniform sampler2D image;
    // uniform vec4 color;

    out vec4 frag_color;

    void main()
    {
      frag_color = vec4(0.4, 0.0, 0.0, 0.8);
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

auto MetricGridRenderer::destroy_opengl_data() -> void
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

    _shader_program.set_uniform_matrix3f("K", _intrinsics.K.data());
    _shader_program.set_uniform_matrix4f("transform",
                                         scene._transform.matrix().data());
    _shader_program.set_uniform_matrix4f("view", scene._view.data());
    _shader_program.set_uniform_matrix4f("projection",
                                         scene._projection.data());
  }

  glBindVertexArray(_vao);
  glDrawElements(GL_TRIANGLES, _triangles.size(), GL_UNSIGNED_INT, 0);
}
