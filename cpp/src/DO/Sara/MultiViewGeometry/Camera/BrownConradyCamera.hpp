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

#include <DO/Sara/Core/Pixel/PixelTraits.hpp>

#include <DO/Sara/ImageProcessing/Interpolation.hpp>

#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/PolynomialDistortionModel.hpp>


namespace DO::Sara {

  template <typename T, typename DistortionModel>
  struct BrownConradyCamera: PinholeCamera<T>
  {
    static constexpr auto eps = static_cast<T>(1e-8);

    using distortion_model_type = DistortionModel;

    //! @brief Types.
    using base_type = PinholeCamera<T>;
    using vector2_type = typename base_type::vector2_type;
    using vector3_type = typename base_type::vector3_type;
    using matrix2_type = Eigen::Matrix<T, 2, 2>;
    using matrix3_type = typename base_type::matrix3_type;

    using base_type::image_sizes;
    using base_type::K;
    using base_type::K_inverse;

    // Distortion model (can be centered, decentered, polynomial, division).
    distortion_model_type distortion_model;

    inline auto undistort(const vector2_type& xd)  const -> vector2_type
    {
      // Normalized coordinates.
      const vector2_type xdn = (K_inverse * xd.homogeneous()).head(2);
      const auto xun  = distortion_model.correct(xdn);
      return (K * xun.homogeneous()).head(2);
    }

    inline auto distort(const vector2_type& xd) const -> vector2_type
    {
      // Normalized coordinates.
      const vector2_type xdn = (K_inverse * xd.homogeneous()).head(2);

      const vector2_type xun = distortion_model.distort(xdn);

      // Go back to pixel coordinates.
      const vector2_type xu = (K * xun.homogeneous()).head(2);

      return xu;
    }

    inline auto project(const vector3_type& x) const -> vector2_type
    {
      const Eigen::Vector2f pixel_coords = (K * x).hnormalized();
      return distort(pixel_coords);
    }

    inline auto backproject(const vector2_type& x) const -> vector3_type
    {
      const Eigen::Vector2f &xu = undistort (x);
      return K_inverse * xu.homogeneous ();
    }
  };


  template <typename T>
  using BrownConradyCamera22 =
      BrownConradyCamera<T, PolynomialDistortionModel<T, 2, 2>>;
  template <typename T>
  using BrownConradyCamera32 =
      BrownConradyCamera<T, PolynomialDistortionModel<T, 3, 2>>;

  template <typename T>
  using BrownConradyCameraDecentered22 =
      BrownConradyCamera<T, DecenteredPolynomialDistortionModel<T, 2, 2>>;

  template <typename T>
  using BrownConradyCameraDecentered32 =
      BrownConradyCamera<T, DecenteredPolynomialDistortionModel<T, 3, 2>>;

}  // namespace DO::Sara
