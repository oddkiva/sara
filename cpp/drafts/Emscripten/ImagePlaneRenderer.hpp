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

#pragma once

#include <DO/Sara/Core.hpp>

#include <drafts/OpenCL/GL.hpp>

#include <memory>


struct ImagePlaneRenderer
{
  struct ImageTexture
  {
    // Initialize the projection matrix once for all.
    Eigen::Matrix4f _projection;
    Eigen::Matrix4f _model_view;
    Eigen::Vector2f _image_sizes;

    DO::Sara::GL::Texture2D _texture;
    GLuint _texture_unit;

    auto set_texture(const DO::Sara::ImageView<DO::Sara::Rgb8>& image,
                     GLuint texture_unit) -> void;

    auto destroy() -> void;
  };

  // OpenGL/Device geometry data.
  DO::Sara::GL::VertexArray _vao;
  DO::Sara::GL::Buffer _vbo;
  DO::Sara::GL::Buffer _ebo;
  // The list of image textures.
  std::vector<ImageTexture> _textures;

  // OpenGL shaders.
  DO::Sara::GL::Shader _vertex_shader;
  DO::Sara::GL::Shader _fragment_shader;
  DO::Sara::GL::ShaderProgram _shader_program;
  // Uniform locations.
  GLint _model_view_loc;
  GLint _projection_loc;
  GLint _image_sizes_loc;
  GLint _image_loc;


  // The singleton.
  static std::unique_ptr<ImagePlaneRenderer> _instance;

  static auto instance() -> ImagePlaneRenderer&;

  auto initialize() -> void;

  auto destroy_gl_objects() -> void;

  auto render(const ImagePlaneRenderer::ImageTexture& image) -> void;
};
