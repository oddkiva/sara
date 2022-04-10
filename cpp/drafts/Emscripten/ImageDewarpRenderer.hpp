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

#include "ImagePlaneRenderer.hpp"

#include <DO/Sara/MultiViewGeometry/Camera/OmnidirectionalCamera.hpp>

#include <drafts/OpenCL/GL.hpp>

#include <memory>


struct ImageDewarpRenderer
{
  struct CameraParameters
  {
    //! @brief The extrinsic camera matrix as a uniform variables.
    Eigen::Matrix3f K;
    Eigen::Matrix3f K_inverse;
    Eigen::Matrix3f R;

    //! @brief The source intrinsic camera parameters.
    DO::Sara::OmnidirectionalCamera<float> _intrinsics;
  };

  // OpenGL Shader program.
  DO::Sara::GL::Shader _vertex_shader;
  DO::Sara::GL::Shader _fragment_shader;
  DO::Sara::GL::ShaderProgram _shader_program;

  // OpenGL geometry data.
  DO::Sara::GL::VertexArray _vao;
  DO::Sara::GL::Buffer _vbo;
  DO::Sara::GL::Buffer _ebo;

  static std::unique_ptr<ImageDewarpRenderer> _instance;

  static auto instance() -> ImageDewarpRenderer&;

  auto initialize() -> void;

  auto destroy_gl_objects() -> void;

  auto render(const ImagePlaneRenderer::ImageTexture&, const CameraParameters&)
      -> void;
};
