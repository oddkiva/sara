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

#include "Scene.hpp"

#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>


namespace sara = DO::Sara;


std::unique_ptr<Scene> Scene::_scene = nullptr;


auto Scene::instance() -> Scene&
{
  if (_scene.get() == nullptr)
    _scene.reset(new Scene{});
  return *_scene;
}

auto Scene::initialize(const sara::ImageView<sara::Rgb8>& image_view) -> void
{
  // Create a vertex shader.
  const std::map<std::string, int> arg_pos = {{"in_coords", 0},      //
                                              {"in_color", 1},       //
                                              {"in_tex_coords", 2},  //
                                              {"out_color", 0}};

  const auto vertex_shader_source = R"shader(#version 300 es
    layout (location = 0) in vec3 in_coords;
    layout (location = 1) in vec3 in_color;
    layout (location = 2) in vec2 in_tex_coords;

    uniform mat4 transform;
    uniform mat4 view;
    uniform mat4 projection;

    out vec2 out_tex_coords;

    void main()
    {
      gl_Position = projection * view * transform * vec4(in_coords, 1.0);
      out_tex_coords = in_tex_coords;
    }
    )shader";
  _vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);

  // Create a fragment shader.
  const auto fragment_shader_source = R"shader(#version 300 es
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

  // Read the image from the disk.
  auto image = sara::Image<sara::Rgb8>{image_view};
  // Flip vertically so that the image data matches OpenGL image coordinate
  // system.
  sara::flip_vertically(image);

  // Copy the image to the GPU texture.
  _texture.generate();
  _texture.bind();
  _texture.set_border_type(GL_CLAMP_TO_EDGE);
  _texture.set_interpolation_type(GL_LINEAR);
  _texture.initialize_data(image, 0);

  // Encode the vertex data in a tensor.
  _vertices = sara::Tensor_<float, 2>{{4, 8}};
  // clang-format off
  _vertices.flat_array() <<
    // coords            color              texture coords
     0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,  // bottom-right
     0.5f,  0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 1.0f,  // top-right
    -0.5f,  0.5f, 0.0f,  1.0f, 1.0f, 0.0f,  0.0f, 1.0f,  // top-left
    -0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f;  // bottom-left
  // clang-format on

  // Resize the quad vertices with the appropriate image ratio.
  const auto image_ratio = static_cast<float>(image.width()) / image.height();
  _vertices.matrix().leftCols(1) *= image_ratio;

  _triangles.resize(2, 3);
  // clang-format off
  _triangles.flat_array() <<
    0, 1, 2,
    2, 3, 0;
  // clang-format on

  const auto row_bytes = [](const sara::TensorView_<float, 2>& data) {
    return data.size(1) * sizeof(float);
  };
  const auto float_pointer = [](int offset) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  _vao.generate();
  // Vertex attributes.
  _vbo.generate();
  // Triangles data.
  _ebo.generate();

  {
    glBindVertexArray(_vao);

    // Copy vertex data.
    _vbo.bind_vertex_data(_vertices);

    // Copy geometry data.
    _ebo.bind_triangles_data(_triangles);

    // Map the parameters to the argument position for the vertex shader.
    //
    // Vertex coordinates.
    glVertexAttribPointer(arg_pos.at("in_coords"), 3 /* 3D points */, GL_FLOAT,
                          GL_FALSE, row_bytes(_vertices), float_pointer(0));
    glEnableVertexAttribArray(arg_pos.at("in_coords"));

    // Colors.
    glVertexAttribPointer(arg_pos.at("in_color"), 3 /* 3D colors */, GL_FLOAT,
                          GL_FALSE, row_bytes(_vertices), float_pointer(3));
    glEnableVertexAttribArray(arg_pos.at("in_color"));

    // Texture coordinates.
    glVertexAttribPointer(arg_pos.at("in_tex_coords"), 2 /* 3D colors */,
                          GL_FLOAT, GL_FALSE, row_bytes(_vertices),
                          float_pointer(6));
    glEnableVertexAttribArray(arg_pos.at("in_tex_coords"));
  }

  _projection.setIdentity();
  _view.setIdentity();
  _transform.setIdentity();
}

auto Scene::destroy_opengl_data() -> void
{
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();

  _vao.destroy();
  _vbo.destroy();
  _ebo.destroy();
  _texture.destroy();
}

auto Scene::render() -> void
{
  _shader_program.use(true);

  // Reset the uniforms.
  {
    // Projection-Model-View matrices.
    _shader_program.set_uniform_matrix4f("transform",
                                         _transform.matrix().data());
    _shader_program.set_uniform_matrix4f("view", _view.data());
    _shader_program.set_uniform_matrix4f("projection", _projection.data());

    // Texture.
    // const auto tex_location = glGetUniformLocation(_shader_program, "image");
    // if (tex_location == GL_INVALID_VALUE)
    //   throw std::runtime_error{"Cannot find texture location!"};
    // glUniform1i(tex_location, 0);
    _shader_program.set_uniform_texture("image", _texture);
  }

  glBindVertexArray(_vao);
  glDrawElements(GL_TRIANGLES, _triangles.size(), GL_UNSIGNED_INT, 0);
}
