#pragma once

#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>

#include <optional>


namespace DO::Sara {

  // @brief The fisheye camera model is defined for field of views of up to 180
  // degrees.
  template <typename T>
  struct FisheyeCamera : PinholeCamera<T>
  {
    static constexpr auto eps = static_cast<T>(1e-8);

    using base_type = PinholeCamera<T>;
    using base_type::image_sizes;
    using base_type::K;
    using base_type::K_inverse;

    //! @brief Types.
    using Vec2 = typename base_type::Vec2;
    using Vec3 = typename base_type::Vec3;

    //! @brief The distortion coefficients.
    Eigen::Matrix<T, 4, 1> k;

    //! @brief Cached table that maps undistorted coordinates to distorted
    //! coordinates.
    std::optional<Image<Eigen::Vector2f>> from_undistorted_to_distorted_coords;
    //! @brief Cached table that maps distorted coordinates to undistorted
    //! coordinates.
    std::optional<Image<Eigen::Vector2f>> from_distorted_to_undistorted_coords;

    //! @brief Iterative method based on the first order Taylor approximation.
    /*!
     *  We can consult the details in:
     *  Geometric camera calibration using circular control points
     *  Janne Heikkila
     *
     *  Equation (7) and (8) provide the insights.
     */
    auto undistort(const Vec2& xd, int num_iterations = 10) const -> Vec2
    {
      // Get the normalized coordinates from the pixel coordinates.
      if (!K_inverse.has_value())
        base_type::cache_inverse_calibration_matrix();
      const Vec2 xd_normalized = (K_inverse.value() * xd.homogeneous()).head(2);

      // Calculate the radial component r of the normalized(r, z), where z = 1.
      auto theta_distorted = xd_normalized.norm();

      // Return NaN values for theta larger than pi/2.
      //
      // In OpenCV, values are clamped instead, which is actually questionable
      // from a 3D reconstruction point of view.
      constexpr auto half_pi = static_cast<T>(M_PI_2);
      if (std::abs(theta_distorted) > half_pi)
        return Vec2{std::numeric_limits<T>::quiet_NaN(),
                    std::numeric_limits<T>::quiet_NaN()};


      // If we are close to the optical center, nothing to do.
      if (theta_distorted < eps)
        return xd;

      auto theta = theta_distorted;
      for (auto iter = 0; iter < num_iterations; ++iter)
      {
        const auto theta2 = theta * theta;
        const auto theta4 = theta2 * theta2;
        const auto theta6 = theta4 * theta2;
        const auto theta8 = theta4 * theta4;
        theta = theta_distorted / (1 + k(0) * theta2 + k(1) * theta4 +
                                   k(2) * theta6 + k(3) * theta8);
      }

      // Sanity check.
      const auto theta_sign_flipped = theta_distorted * theta < 0;
      if (theta_sign_flipped)
        return Vec2{std::numeric_limits<T>::quiet_NaN(),
                    std::numeric_limits<T>::quiet_NaN()};


      const auto undistortion_factor = std::tan(theta) / theta_distorted;
      const Vec2 xu_normalized = xd_normalized * undistortion_factor;
      const Vec2 xu = (K * xu_normalized.homogeneous()).head(2);

      return xu;
    }

    auto distort(const Vec2& xu) const -> Vec2
    {
      // Calculate the normalized coordinates.
      if (!K_inverse.has_value())
        base_type::cache_inverse_calibration_matrix();
      const Vec2 xun = (K_inverse.value() * xu.homogeneous()).head(2);

      // Calculate the radial component r of the equivalent cylindric
      // coordinates (r, z) of the normalized coordinates, in which case z = 1.
      const auto r = xun.norm();

      // Check this edge case: if we are at the center of the image, there is no
      // distortion.
      if (r < eps)
        return xu;

      // Otherwise we treat the general case.
      //
      // 1. Calculate the powers of theta.
      const auto theta = std::atan(r);
      const auto theta2 = theta * theta;
      const auto theta3 = theta2 * theta;
      const auto theta5 = theta3 * theta2;
      const auto theta7 = theta5 * theta2;
      const auto theta9 = theta7 * theta2;

      // 2. Invert the radial distance to the optical center.
      const auto r_inverse = 1 / r;

      // 3. Calculate the distortion factor.
      auto distortion_factor = r_inverse * (theta +          //
                                            k(0) * theta3 +  //
                                            k(1) * theta5 +  //
                                            k(2) * theta7 +  //
                                            k(3) * theta9);

      // Rescale the radial component with the distortion factor.
      const Vec2 xdn = xun * distortion_factor;

      // Finally go back to the pixel coordinates.
      const Vec2 xd = (K * xdn.homogeneous()).head(2);

      return xd;
    }

    inline auto project(const Vec3& x) const -> Vec2
    {
      return distort(base_type::project(x));
    }

    inline auto backproject(const Vec2& x) const -> Vec3
    {
      base_type::backproject(undistort(x));
    }
  };

}  // namespace DO::Sara
