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

#include <DO/Kalpana/EasyGL.hpp>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

  struct ColoredPointCloud
  {
    static constexpr auto coords_dim = 3;
    static constexpr auto color_dim = 3;
    static constexpr auto coords_attr_index = 0;
    static constexpr auto color_attr_index = 1;
    static constexpr auto vertex_dim = coords_dim + color_dim;

    // Point cloud data layout specified for OpenGL.
    VertexArray _vao;
    // 3D points and other attributes (colors, normals, etc).
    Buffer _vbo;
    // Point cloud data sizes
    Eigen::Vector2i _sizes;

    // Initialize the VAO and VBO.
    auto initialize() -> void;

    auto destroy() -> void;

    auto upload_host_data_to_gl(const Sara::TensorView_<float, 2>& point_cloud)
        -> void;
  };

  //! @}

}  // namespace DO::Kalpana::GL
