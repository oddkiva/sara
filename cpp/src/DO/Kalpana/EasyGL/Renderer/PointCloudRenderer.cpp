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

#include <DO/Kalpana/EasyGL/Renderer/PointCloudRenderer.hpp>


using namespace DO::Kalpana::GL;

auto PointCloudRenderer::initialize() -> void
{
  const auto vertex_shader_source = R"shader(#version 300 es
    layout (location = 0) in vec3 in_coords;

    uniform float point_size;
    uniform mat4 transform;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 out_color;

    void main()
    {
      gl_Position = projection * view * transform * vec4(in_coords, 1.0);
      gl_PointSize = point_size;
      out_color = vec3(1.0, 0.5, 0.5);
    }
  )shader";

  const auto fragment_shader_source = R"shader(#version 300 es
    #ifdef GL_ES
    precision highp float;
    #endif

    in vec3 out_color;
    out vec4 frag_color;

    void main()
    {
      vec2 m = gl_PointCoord - 0.5;
      float squared_dist = dot(m, m);
      // Signed distance function of a circle of radius 0.5.
      if (squared_dist > 0.25)
        discard;
      float alpha = smoothstep(0.0, 0.1, 1. - 2. * sqrt(squared_dist));

      frag_color = vec4(out_color, alpha);
    }
  )shader";

  this->initialize(vertex_shader_source, fragment_shader_source);
}

auto PointCloudRenderer::initialize(const std::string& vertex_shader_source,
                                    const std::string& fragment_shader_source)
    -> void
{
  _vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);
  _fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                      fragment_shader_source);

  _shader_program.create();
  _shader_program.attach(_vertex_shader, _fragment_shader);

#ifndef __EMSCRIPTEN__
  _shader_program.use();
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();
#endif
}

auto PointCloudRenderer::destroy() -> void
{
  _shader_program.detach();
  _shader_program.clear();
}

auto PointCloudRenderer::render(const ColoredPointCloud& point_cloud,
                                const float point_size,
                                const Eigen::Matrix4f& model_view,
                                const Eigen::Matrix4f& projection) -> void
{
  _shader_program.use();
  _shader_program.set_uniform_param("point_size", point_size);

  // View matrix.
  _shader_program.set_uniform_matrix4f("view", model_view.data());

  // Projection matrix.
  _shader_program.set_uniform_matrix4f("projection", projection.data());

  // Draw triangles.
  glBindVertexArray(point_cloud._vao);
  const auto& num_points = static_cast<GLsizei>(point_cloud._sizes.x());
  glDrawArrays(GL_POINTS, 0, num_points);
}
