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

#pragma once

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Kalpana/EasyGL/Buffer.hpp>
#include <DO/Kalpana/EasyGL/OpenGL.hpp>
#include <DO/Kalpana/EasyGL/Shader.hpp>
#include <DO/Kalpana/EasyGL/VertexArray.hpp>


namespace DO::Kalpana::GL {

  struct CheckerboardRenderer
  {
    CheckerboardRenderer(const int rows = 20, const int cols = 20,
                         const float scale = 10.f, const float y_origin = -1.f);

    auto initialize_geometry() -> void;

    auto initialize_shader_program() -> void;

    auto initialize() -> void;

    auto destroy() -> void;

    auto render(const Eigen::Matrix4f& transform,
                const Eigen::Matrix4f& model_view,
                const Eigen::Matrix4f& projection) -> void;

    int _rows;
    int _cols;
    Sara::Tensor_<float, 2> _vertices;
    Sara::Tensor_<std::uint32_t, 2> _triangles;
    Buffer _vbo;
    Buffer _ebo;
    VertexArray _vao;

    Shader _vertex_shader;
    Shader _fragment_shader;
    ShaderProgram _shader_program;
    GLint _transform_loc;
    GLint _view_loc;
    GLint _projection_loc;
  };

}  // namespace DO::Kalpana::GL
