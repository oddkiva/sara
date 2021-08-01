// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <Eigen/Dense>


namespace DO::Sara {

  template <typename T>
  struct PinholeCamera
  {
    //! @brief Types.
    using scalar_type = T;
    using vector2_type = Eigen::Matrix<T, 2, 1>;
    using vector3_type = Eigen::Matrix<T, 3, 1>;
    using matrix3_type = Eigen::Matrix<T, 3, 3>;

    //! @brief Original image sizes by the camera.
    vector2_type image_sizes;
    //! @brief Pinhole camera parameters.
    matrix3_type K;

    //! @brief Inverse calibration matrix.
    matrix3_type K_inverse;

    //! @brief Calculate and cache the inverse of the calibration matrix.
    inline auto calculate_inverse_calibration_matrix() -> void
    {
      K_inverse = K.inverse();
    }

    inline auto focal_lengths() const -> vector2_type
    {
      return {K(0, 0), K(1, 1)};
    }

    inline auto principal_point() const -> vector2_type
    {
      return K.col(2).head(2);
    }

    inline auto field_of_view() const -> vector2_type
    {
      const Eigen::Array<T, 2, 1> tg = image_sizes.array() /  //
                                       focal_lengths().array() / 2.;
      return 2. * tg.atan();
    }

    inline auto project(const vector3_type& x) const -> vector2_type
    {
      return (K * x).hnormalized();
    }

    inline auto backproject(const vector2_type& x) const -> vector3_type
    {
      return K_inverse * x.homogeneous();
    }

    inline auto downscale_image_sizes(T factor) -> void
    {
      image_sizes /= factor;
      K.block(0, 0, 2, 3) /= factor;
      K_inverse = K.inverse();
    }
  };

}  // namespace DO::Sara
