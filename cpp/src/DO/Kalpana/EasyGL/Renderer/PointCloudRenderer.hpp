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

#include <DO/Kalpana/EasyGL.hpp>
#include <DO/Kalpana/EasyGL/Objects/PointCloud.hpp>

#include <string>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

  struct PointCloudRenderer
  {
    //! @brief Shader data structure are understood as optional values.
    //! @{
    Shader _vertex_shader;
    Shader _fragment_shader;
    //! @}

    //! The shader program is understood as an optional value.
    ShaderProgram _shader_program;

    auto initialize() -> void;

    auto initialize(const std::string& vertex_shader_source,
                    const std::string& fragment_shader_source) -> void;

    auto destroy() -> void;

    auto render(const ColoredPointCloud& point_cloud, const float point_size,
                const Eigen::Matrix4f& model_view,
                const Eigen::Matrix4f& projection) -> void;
  };

  //! @}

}  // namespace DO::Kalpana::GL
