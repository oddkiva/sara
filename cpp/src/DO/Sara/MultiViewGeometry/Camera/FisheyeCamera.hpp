#pragma once

#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>

#include <optional>


namespace DO::Sara {

  template <typename T>
  struct FisheyeCamera : PinholeCamera<T>
  {
    using base_type = PinholeCamera<T>;
    using base_type::K;
    using base_type::K_inverse;

    //! @brief Types.
    using Vec2 = typename base_type::Vec2;
    using Vec3 = typename base_type::Vec3;

    //! @brief The distortion coefficients.
    Eigen::Matrix<T, Eigen::Dynamic, 1> k;

    //! @brief Cached table that maps undistorted coordinates to distorted
    //! coordinates.
    std::optional<Image<Eigen::Vector2f>> from_undistorted_to_distorted_coords;
    //! @brief Cached table that maps distorted coordinates to undistorted
    //! coordinates.
    std::optional<Image<Eigen::Vector2f>> from_distorted_to_undistorted_coords;

    inline auto focalLengths() const -> Vec2
    {
      return {K(0, 0), K(1, 1)};
    }

    inline auto principalPoint() const -> Vec2
    {
      return K.col(2).head(2);
    }

    auto undistort(const Vec2&) const -> Vec2
    {
      throw std::runtime_error{"Not implemented"};
      return {};
    }

    auto distort(const Vec2& xu) const -> Vec2
    {
      static constexpr auto eps = static_cast<T>(1e-8);

      // Calculate the normalized coordinates.
      if (!K_inverse.has_value())
        base_type::calculate_inverse_calibration_matrix();
      const Vec2 xun = (K_inverse.value() * xu.homogeneous()).head(2);

      // Calculate the radial component r of the equivalent cylindric
      // coordinates (r, z) of the normalized coordinates, in which case z = 1.
      const auto r = xun.norm();

      // Check this edge case: if we are at the center of the image, there is no
      // distortion.
      if (r > eps)
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
  };

}  // namespace DO::Sara
