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

#include <DO/Sara/Core/Math/Rotation.hpp>


namespace DO::Sara {

  //! @brief We follow the automotive axes description for 3D coordinates.
  enum class AxisConvention : std::uint8_t
  {
    Camera = 0,
    ComputerVision = 0,
    OpenGL = 1,
    Automotive = 2,
    Aerospace = 3,
    Naval = 3,
  };


  //! @brief Permutation matrix to go from the camera axis convention to the
  //! target axis convention.
  inline auto axis_permutation_matrix(AxisConvention convention)
      -> Eigen::Matrix3i
  {
    // By default no permutation.
    Eigen::Matrix3i P = Eigen::Matrix3i::Identity();

    // clang-format off
    if (convention == AxisConvention::Automotive) {
      P <<
         0,  0, 1, // Camera Z =          Automotive X
        -1,  0, 0, // Camera X = Negative Automotive Y
         0, -1, 0; // Camera Y = Negative Automotive Z
    }
    else if (convention == AxisConvention::Aerospace || //
             convention == AxisConvention::Naval) {
      P <<
        0, 0, 1, // Camera Z = Automotive X
        1, 0, 0, // Camera X = Automotive Y
        0, 1, 0; // Camera Y = Automotive Z

    }
    else if (convention == AxisConvention::OpenGL) {
      P <<
        1,  0,  0,
        0, -1,  0,
        0,  0, -1;
    }
    // clang-format on

    return P;
  }

  template <AxisConvention convention, typename T>
  inline auto equivalent_intrinsic_orientation(
      const Eigen::Matrix<T, 3, 3>& camera_frame_orientation)
      -> Eigen::Matrix<T, 3, 3>
  {
    static const Eigen::Matrix<T, 3, 3> P =
        axis_permutation_matrix(convention)  //
            .template cast<T>()
            .transpose();
    return P * camera_frame_orientation;
  }


}  // namespace DO::Sara
