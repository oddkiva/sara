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

#include <DO/Sara/MultiViewGeometry/Camera/OmnidirectionalCamera.hpp>

#include <drafts/OpenCL/GL.hpp>

#include <Eigen/Geometry>

#include <memory>


struct MetricGridRenderer
{
  //! @brief The extrinsic camera parameters.
  Eigen::Matrix4f _extrinsics;
  //! @brief The intrinsic camera parameters.
  DO::Sara::OmnidirectionalCamera<float> _intrinsics;

  //! @brief Line geometry data.
  //!
  //! A line can be thought a "thick" rectangular line. This thick line can then
  //! be thought as a quad.
  //!
  //! In turn this quad can be decomposed into two right triangles adjacent at
  //! their hypotenuse.
  std::vector<float> _vertices;
  std::vector<std::uint32_t> _triangles;

  // OpenGL geometry data.
  DO::Sara::GL::VertexArray _vao;
  DO::Sara::GL::Buffer _vbo;
  DO::Sara::GL::Buffer _ebo;

  // OpenGL Shader program.
  DO::Sara::GL::Shader _vertex_shader;
  DO::Sara::GL::Shader _fragment_shader;
  DO::Sara::GL::ShaderProgram _shader_program;

  static std::unique_ptr<MetricGridRenderer> _renderer;

  static auto instance() -> MetricGridRenderer&;

  auto add_line_segment(const Eigen::Vector2f& a,  //
                        const Eigen::Vector2f& b,  //
                        float thickness = 0.02f,   //
                        float antialias_radius = 0.01f) -> void;

  auto initialize() -> void;

  auto transfer_line_tesselation_to_gl_buffers() -> void;

  auto destroy_gl_objects() -> void;

  auto render() -> void;
};
