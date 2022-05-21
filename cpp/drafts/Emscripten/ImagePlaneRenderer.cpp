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

#include "ImagePlaneRenderer.hpp"

#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>


namespace sara = DO::Sara;


auto ImagePlaneRenderer::ImageTexture::set_texture(
    const sara::ImageView<sara::Rgb8>& image_view, GLuint texture_unit) -> void
{
  // Specify the image sizes.
  _image_sizes << image_view.width(), image_view.height();

  // Flip vertically so that the image data matches OpenGL image coordinate
  // system.
  auto image = sara::Image<sara::Rgb8>{image_view};

  // Bind the texture unit: GL_TEXTURE0 + i.
  _texture_unit = texture_unit;
  glActiveTexture(GL_TEXTURE0 + _texture_unit);

  // Initialize the texture object and bind it.
  //
  // The texture unit GL_TEXTURE0 + i will be associated to this texture object.
  if (!_texture)
    _texture.generate();
  _texture.bind();

  // Copy the image to the GPU texture.
  _texture.initialize_data(image, 0);

  // Set the image display options.
  _texture.set_border_type(GL_CLAMP_TO_EDGE);
  _texture.set_interpolation_type(GL_LINEAR);
}

auto ImagePlaneRenderer::ImageTexture::destroy() -> void
{
  _texture.destroy();
}


std::unique_ptr<ImagePlaneRenderer> ImagePlaneRenderer::_instance = nullptr;


auto ImagePlaneRenderer::instance() -> ImagePlaneRenderer&
{
  if (_instance.get() == nullptr)
    _instance.reset(new ImagePlaneRenderer{});
  return *_instance;
}

auto ImagePlaneRenderer::initialize() -> void
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
      // Change the aspect ratio of the square so that it conforms with the
      // image one.
      float image_ratio = image_sizes.x / image_sizes.y;
      vec2 final_coords = vec2(in_coords.x * image_ratio, in_coords.y);

      gl_Position = projection * model_view * vec4(final_coords, 0.0, 1.0);
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
  _image_sizes_loc = _shader_program.get_uniform_location("image_sizes");
  _image_loc = _shader_program.get_uniform_location("image");

#ifndef __EMSCRIPTEN__
  // Clearing the shaders after attaching them to the shader program does not
  // work on WebGL 2.0/OpenGL ES 3.0... I don't know why.
  _shader_program.use();
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();
#endif

  // Generate a single square geometry data. This data will be constantly reused
  // to display any image on the screen.
  //
  // Encode the vertex data of the square in a tensor.
  auto vertices = sara::Tensor_<float, 2>{{4, 5}};
  // clang-format off
  vertices.flat_array() <<
  // coords              texture coords
     0.5f, -0.5f, 0.0f,  1.0f, 1.0f,  // bottom-right
     0.5f,  0.5f, 0.0f,  1.0f, 0.0f,  // top-right
    -0.5f,  0.5f, 0.0f,  0.0f, 0.0f,  // top-left
    -0.5f, -0.5f, 0.0f,  0.0f, 1.0f;  // bottom-left
  // clang-format on

  auto triangles = sara::Tensor_<std::uint32_t, 2>{{2, 3}};
  // clang-format off
  triangles.flat_array() <<
    0, 1, 2,
    2, 3, 0;
  // clang-format on

  if (!_vao)
    _vao.generate();
  if (!_vbo)
    _vbo.generate();
  if (!_ebo)
    _ebo.generate();

  glBindVertexArray(_vao);

  // Copy vertex data.
  _vbo.bind_vertex_data(vertices);

  // Copy geometry data.
  _ebo.bind_triangles_data(triangles);

  static constexpr auto row_bytes =
      [](const sara::TensorView_<float, 2>& data) {
        return data.size(1) * sizeof(float);
      };
  static constexpr auto float_pointer = [](int offset) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  // Map the parameters to the argument position for the vertex shader.
  //
  // Vertex coordinates.
  glVertexAttribPointer(                           //
      arg_pos.at("in_coords"), 3 /* 3D points */,  //
      GL_FLOAT, GL_FALSE, row_bytes(vertices), float_pointer(0));
  glEnableVertexAttribArray(arg_pos.at("in_coords"));

  // Texture coordinates.
  glVertexAttribPointer(
      arg_pos.at("in_tex_coords"), 2 /* texture coordinates */,  //
      GL_FLOAT, GL_FALSE, row_bytes(vertices), float_pointer(3));
  glEnableVertexAttribArray(arg_pos.at("in_tex_coords"));
}

auto ImagePlaneRenderer::destroy_gl_objects() -> void
{
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();

  _vao.destroy();
  _vbo.destroy();
  _ebo.destroy();
}

auto ImagePlaneRenderer::render(const ImagePlaneRenderer::ImageTexture& image)
    -> void
{
  _shader_program.use(true);

  // Set the uniforms.
  //
  // Projection-Model-View matrices.
  _shader_program.set_uniform_matrix4f(_model_view_loc,
                                       image._model_view.data());
  _shader_program.set_uniform_matrix4f(_projection_loc,
                                       image._projection.data());
  // Image ratio.
  _shader_program.set_uniform_vector2f(_image_sizes_loc,
                                       image._image_sizes.data());
  _shader_program.set_uniform_texture(_image_loc, image._texture_unit);

  glBindVertexArray(_vao);

  static constexpr auto triangle_index_count = 6;
  glDrawElements(GL_TRIANGLES, triangle_index_count, GL_UNSIGNED_INT, 0);
}
