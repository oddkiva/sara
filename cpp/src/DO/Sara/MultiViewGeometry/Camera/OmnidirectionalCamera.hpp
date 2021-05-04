#pragma once

#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>

#include <optional>


namespace DO::Sara {

  template <typename T>
  struct OmnidirectionalCamera : PinholeCamera<T>
  {
    static constexpr auto eps = static_cast<T>(1e-8);

    using base_type = PinholeCamera<T>;
    using base_type::K;
    using base_type::K_inverse;

    //! @brief Types.
    using Vec2 = typename base_type::Vec2;
    using Vec3 = typename base_type::Vec3;

    //! @brief The distortion coefficients.
    Eigen::Matrix<T, 3, 1> radial_distortion_coefficients;
    Eigen::Matrix<T, 3, 1> tangential_distortion_coefficients;
    T xi;

    //! @brief Cached table that maps undistorted coordinates to distorted
    //! coordinates.
    std::optional<Image<Eigen::Vector2f>> from_undistorted_to_distorted_coords;
    //! @brief Cached table that maps distorted coordinates to undistorted
    //! coordinates.
    std::optional<Image<Eigen::Vector2f>> from_distorted_to_undistorted_coords;

    //! @brief Apply only in the normalized coordinates.
    auto distortion_delta(const Vec2& m_undistorted) const -> Vec2
    {
      const auto& m = m_undistorted;

      // Distortion.
      const auto& k1 = radial_distortion_coefficients(0);
      const auto& k2 = radial_distortion_coefficients(1);
      const auto& k3 = radial_distortion_coefficients(2);
      const auto& p1 = tangential_distortion_coefficients(0);
      const auto& p2 = tangential_distortion_coefficients(1);

      // Radial component (additive).
      const auto r2 = m.squaredNorm();
      const auto r4 = r2 * r2;
      const auto r6 = r4 * r2;
      const Vec2 radial_factor = m * (k1 * r2 + k2 * r4 + k3 * r6);

      // Tangential component (additive).
      const auto tx = 2 * p1 * m.x() * m.y() + p2 * (r2 + 2 * p1 * m.x());
      const auto ty = p1 * (r2 + 2 * p1 * m.y()) + 2 * p2 * m.x() * m.y();

      // Apply the distortion.
      const Vec2 delta = radial_factor + Vec2{tx, ty};

      return delta;
    }

    //! @brief Project a 3D scene point expressed in the camera frame to the
    //! distorted image coordinates.
    auto project(const Vec3& x) const -> Vec2
    {
      // Mirror transformation
      //
      // 1. Project on the unit sphere (reflection from the spherical mirror).
      const Vec3 xs = x.normalized();
      // 2. Change coordinates.
      const Vec3 xe = xs + xi * Vec3::UnitZ();
      // 3. Project the reflected ray by the mirror to the normalized plane z = 1.
      const Vec2 m = xe.hnormalized();

      // Apply the distortion.
      const Vec2 m_distorted = m + distortion_delta(m);

      // Go back to pixel coordinates.
      const Vec2 p = (K * m_distorted.homogeneous()).head(2);

      return p;
    }

    //! Iterative method to remove distortion.
    auto undistort(const Vec2& pd, int num_iterations = 10) const -> Vec2
    {
      // Calculate the normalized coordinates.
      if (!K_inverse.has_value())
        base_type::cache_inverse_calibration_matrix();
      // Normalized distorted coordinates.
      const Vec2 md = (K_inverse.value() * pd.homogeneous()).head(2);

      auto mu = md;
      for (auto iter = 0; iter < num_iterations &&
                          (mu + distortion_delta(mu) - md).norm() > eps;
           ++iter)
      {
        mu = md - distortion_delta(mu);
      }

      return mu;
    }
  };

}  // namespace DO::Sara
