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


struct Scene
{
  // Host geometry data
  DO::Sara::Tensor_<float, 2> _vertices;
  DO::Sara::Tensor_<std::uint32_t, 2> _triangles;

  // OpenGL/Device geometry data.
  DO::Sara::GL::VertexArray _vao;
  DO::Sara::GL::Buffer _vbo;
  DO::Sara::GL::Buffer _ebo;

  // OpenGL shaders.
  DO::Sara::GL::Shader _vertex_shader;
  DO::Sara::GL::Shader _fragment_shader;
  DO::Sara::GL::ShaderProgram _shader_program;
  DO::Sara::GL::Texture2D _texture;

  // Initialize the projection matrix once for all.
  Eigen::Matrix4f _projection;
  Eigen::Matrix4f _model_view;

  Eigen::Vector2f _image_sizes;

  // The singleton.
  static std::unique_ptr<Scene> _scene;

  static auto instance() -> Scene&;

  auto initialize(const DO::Sara::ImageView<DO::Sara::Rgb8>& image) -> void;

  auto destroy_gl_objects() -> void;

  auto render() -> void;
};
