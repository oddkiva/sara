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

#include <optional>


namespace DO::Sara {

  template <typename T>
  struct PinholeCamera
  {
    //! @brief Types.
    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;

    using Mat3 = Eigen::Matrix<T, 3, 3>;

    //! @brief Original image sizes by the camera.
    Vec2 image_sizes;
    //! @brief Pinhole camera parameters.
    Mat3 K;

    //! @brief Cached inverse calibration matrix.
    mutable std::optional<Mat3> K_inverse;

    //! @brief Calculate and cache the inverse of the calibration matrix.
    inline auto cache_inverse_calibration_matrix() const -> void
    {
      K_inverse = K.inverse();
    }

    inline auto focal_lengths() const -> Vec2
    {
      return {K(0, 0), K(1, 1)};
    }

    inline auto principal_point() const -> Vec2
    {
      return K.col(2).head(2);
    }

    inline auto field_of_view() const -> Vec2
    {
      const Eigen::Array<T, 2, 1> tg = image_sizes.array() /  //
                                       focal_lengths().array() / 2.;
      return 2. * tg.atan();
    }

    auto downscale_image_sizes(T factor) -> void
    {
      K.block(0, 0, 2, 3) /= factor;
      image_sizes /= factor;
    }
  };

}
