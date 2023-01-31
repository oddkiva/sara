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

#include <DO/Kalpana/EasyGL/Renderer/TextureRenderer.hpp>

#include <map>


using namespace DO::Kalpana::GL;

auto TextureRenderer::initialize() -> void
{
  // Create a vertex shader.
  static const std::map<std::string, int> arg_pos = {{"in_coords", 0},       //
                                                     {"in_tex_coords", 1}};  //

  static const auto vertex_shader_source = R"shader(#version 300 es
    layout (location = 0) in vec3 in_coords;
    layout (location = 1) in vec2 in_tex_coords;

    uniform mat4 projection;
    uniform mat4 model_view;
    uniform vec2 image_sizes;

    out vec2 out_tex_coords;

    void main()
    {
      gl_Position = projection * model_view * vec4(in_coords, 1.0);
      out_tex_coords = in_tex_coords;
    }
  )shader";
  _vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);

  // Create a fragment shader.
  static const auto fragment_shader_source = R"shader(#version 300 es
    #ifdef GL_ES
    precision highp float;
    #endif

    in vec2 out_tex_coords;
    out vec4 frag_color;

    uniform sampler2D image;

    void main()
    {
      frag_color = texture(image, out_tex_coords);
    }
  )shader";
  _fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                      fragment_shader_source);
  _shader_program.create();
  _shader_program.attach(_vertex_shader, _fragment_shader);

  _model_view_loc = _shader_program.get_uniform_location("model_view");
  _projection_loc = _shader_program.get_uniform_location("projection");
  _image_loc = _shader_program.get_uniform_location("image");

#ifndef __EMSCRIPTEN__
  // Clearing the shaders after attaching them to the shader program does not
  // work on WebGL 2.0/OpenGL ES 3.0... I don't know why.
  _shader_program.use();
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();
#endif
}

auto TextureRenderer::destroy() -> void
{
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();
}

auto TextureRenderer::render(TexturedImage2D& what,
                             TexturedQuad& where,
                             const Eigen::Matrix4f& model_view,
                             const Eigen::Matrix4f& projection) -> void
{
  _shader_program.use(true);

  SARA_CHECK(model_view);
  SARA_CHECK(projection);

  // Set the uniforms.
  //
  // Projection-Model-View matrices.
  _shader_program.set_uniform_matrix4f(_model_view_loc,
                                       const_cast<float*>(model_view.data()));
  _shader_program.set_uniform_matrix4f(_projection_loc,
                                       const_cast<float*>(projection.data()));
  _shader_program.set_uniform_texture(_image_loc, what.texture_unit());

  SARA_CHECK(where.host_vertices().matrix());
  SARA_CHECK(where.host_triangles().matrix());
  SARA_CHECK(where.host_triangles().size());

  // We want to draw the number of triangle indices:
  // 3 vertex indices per triangles x 2 triangles  = 6 vertex indices
  glBindVertexArray(where.vao());
  glDrawElements(GL_TRIANGLES, where.host_triangles().size(), GL_UNSIGNED_INT,
                 0);
}
