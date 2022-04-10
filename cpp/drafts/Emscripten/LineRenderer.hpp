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

#include <drafts/OpenCL/GL.hpp>

#include <memory>


struct LineRenderer
{
  //! A line is implemented a "thick" rectangular line.
  //!
  //! Such a thick line can then be thought as a quad. We decompose this quad
  //! into two right triangles adjacent at their hypotenuse.
  struct LineHostData
  {
    std::vector<float> _vertices;
    std::vector<std::uint32_t> _triangles;

    inline auto clear() -> void
    {
      _vertices.clear();
      _triangles.clear();
    };

    auto add_line_segment(const Eigen::Vector2f& a,  //
                          const Eigen::Vector2f& b,  //
                          float thickness,           //
                          float antialias_radius) -> void;

    auto add_line_segment_in_pixel_coordinates(
        const Eigen::Vector2f& a,            //
        const Eigen::Vector2f& b,            //
        const Eigen::Vector2f& image_sizes,  //
        float thickness = 2.f,               //
        float antialias_radius = 1.f) -> void;
  };

  struct LineGLObjects
  {
    //! We choose one VAO per object.
    DO::Sara::GL::VertexArray _vao;
    DO::Sara::GL::Buffer _vbo;
    DO::Sara::GL::Buffer _ebo;
    Eigen::Vector4f _color;
    std::size_t _triangle_index_count;

    auto set_data(const LineHostData&) -> void;
    auto destroy() -> void;
  };

  // OpenGL Shader program.
  DO::Sara::GL::Shader _vertex_shader;
  DO::Sara::GL::Shader _fragment_shader;
  DO::Sara::GL::ShaderProgram _shader_program;

  // Line data.
  std::vector<LineGLObjects> _lines;

  static std::unique_ptr<LineRenderer> _instance;

  static auto instance() -> LineRenderer&;

  auto initialize() -> void;

  auto destroy_gl_objects() -> void;

  auto render(const ImagePlaneRenderer::ImageTexture&, const LineGLObjects&) -> void;
};
