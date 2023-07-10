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

#include <DO/Kalpana/EasyGL.hpp>

#include <memory>


struct MetricGridRenderer
{
  struct LineHostData
  {
    std::vector<float> _vertices;
    std::vector<std::uint32_t> _triangles;

    inline auto clear() -> void
    {
      _vertices.clear();
      _triangles.clear();
    }

    auto add_line_segment(const Eigen::Vector2f& a,  //
                          const Eigen::Vector2f& b,  //
                          float thickness = 0.02f,   //
                          float antialias_radius = 0.01f) -> void;
  };

  struct LineShaderData
  {
    //! @brief The extrinsic camera matrix as a uniform variables.
    Eigen::Matrix4f _extrinsics;
    //! @brief The intrinsic camera parameters as uniform variables.
    DO::Sara::OmnidirectionalCamera<float> _intrinsics;

    //! @brief OpenGL buffer objects.
    DO::Kalpana::GL::VertexArray _vao;
    DO::Kalpana::GL::Buffer _vbo;
    DO::Kalpana::GL::Buffer _ebo;

    Eigen::Vector4f _color;
    std::size_t _triangle_index_count;

    auto set_data(const LineHostData&) -> void;
    auto destroy() -> void;
  };

  // OpenGL Shader program.
  DO::Kalpana::GL::Shader _vertex_shader;
  DO::Kalpana::GL::Shader _fragment_shader;
  DO::Kalpana::GL::ShaderProgram _shader_program;
  // Uniform locations.
  GLint _image_loc;
  GLint _image_sizes_loc;
  GLint _view_loc;
  GLint _projection_loc;
  // Source camera params locations.
  GLint _C_loc;
  GLint _K_loc;
  GLint _k_loc;
  GLint _p_loc;
  GLint _xi_loc;
  // Line color.
  GLint _color_loc;

  std::vector<LineShaderData> _lines;

  auto initialize() -> void;

  auto destroy_gl_objects() -> void;

  auto render(const ImagePlaneRenderer::ImageTexture&, const LineShaderData&)
      -> void;
};
