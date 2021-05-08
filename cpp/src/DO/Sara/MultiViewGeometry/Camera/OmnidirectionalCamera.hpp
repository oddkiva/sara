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
    Eigen::Matrix<T, 2, 1> radial_distortion_coefficients;
    Eigen::Matrix<T, 2, 1> tangential_distortion_coefficients;
    T xi;

    //! @brief Cached table that maps undistorted coordinates to distorted
    //! coordinates.
    std::optional<Image<Eigen::Vector2f>> from_undistorted_to_distorted_coords;
    //! @brief Cached table that maps distorted coordinates to undistorted
    //! coordinates.
    std::optional<Image<Eigen::Vector2f>> from_distorted_to_undistorted_coords;

    //! @brief Apply only in the normalized coordinates.
    auto distortion_term(const Vec2& m_undistorted) const -> Vec2
    {
      const auto& m = m_undistorted;

      // Distortion.
      const auto& k1 = radial_distortion_coefficients(0);
      const auto& k2 = radial_distortion_coefficients(1);
      const auto& p1 = tangential_distortion_coefficients(0);
      const auto& p2 = tangential_distortion_coefficients(1);

      // Radial component (additive).
      const auto r2 = m.squaredNorm();
      const auto r4 = r2 * r2;
      const Vec2 radial_factor = m * (k1 * r2 + k2 * r4);

      // Tangential component (additive).
      const auto tx = 2 * p1 * m.x() * m.y() + p2 * (r2 + 2 * p1 * m.x());
      const auto ty = p1 * (r2 + 2 * p1 * m.y()) + 2 * p2 * m.x() * m.y();

      // Apply the distortion.
      const Vec2 delta = radial_factor + Vec2{tx, ty};

      return delta;
    }

    //! @brief Project a 3D scene point expressed in the camera frame to the
    //! distorted image coordinates.
    auto project(const Vec3& X) const -> Vec2
    {
      // Mirror transformation
      //
      // 1. Project on the unit sphere (reflection from the spherical mirror).
      const Vec3 Xs = X.normalized();
      // 2. Change coordinates.
      const Vec3 Xe = Xs + xi * Vec3::UnitZ();
      // 3. Project the reflected ray by the mirror to the normalized plane z = 1.
      const Vec2 m = Xe.hnormalized();

      // Apply the distortion.
      const Vec2 m_distorted = m + distortion_term(m);

      // Go back to pixel coordinates.
      const Vec2 p = (K * m_distorted.homogeneous()).head(2);

      return p;
    }

    //! @brief Calculate the inverse of the mirror transformation.
    /*!
     *  The paper terms it as lifting function.
     *
     *  The lifting function is determined by solving a second degree polynomial
     *  in z by exploiting some key algebraic observation:
     *  (1) zs > 0
     *  (2) xs^2 + ys^2 + zs^2 = 1 because the world points are reflected the
     *                             spherical mirror of radius = 1.
     *  (3) xi < 1 by design.
     *
     *  IMPORTANT:
     *  After observing carefully Figure 4 in the paper:
     *  https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
     *  is that by design the lower hemisphere of the mirror can still project
     *  world points behind the omnidirectional camera and if the camera film
     *  plane is wide enough!
     *
     *  Inevitably, zs < 0! Assumption (1) will be violated in this case and the
     *  lifting formula will need to be amended. OUCH!
     *  And this will happen if we know the camera we are dealing with has a FOV
     *  wider than 180 degrees.
     *
     *  Most likely, the authors most likely considered cameras with FOV <= 180
     *  degrees at most and did not think the contrary would happen... So we
     *  will have to address it at some point.
     */
    auto lifting(const Vec2& m) const -> Vec3
    {
      // Calculate auxiliary terms.
      const auto xi_squared = xi * xi;
      const auto m_squared_norm = m.squaredNorm();

      // In the paper: https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
      // Because xi < 1, the discriminant is > 1.
      const auto discriminant = 1 + std::sqrt((1 - xi_squared) * m_squared_norm);

      // Calculate the z-coordinate of Xe.
      const auto numerator = xi + std::sqrt(discriminant);
      // TODO: there *will* be cases where we have to find out whether we have
      // to do this instead for camera with FOV >= 180 degrees.
      // const auto numerator = xi - std::sqrt(discriminant);

      // Determine the z coordinate of the point on the sphere on the decentered
      // reference frame.
      const auto denominator = m_squared_norm + 1;
      const auto ze = numerator / denominator;

      // Back to the reference frame centered in the mirror centre.
      const auto xs = ze * m.x();
      const auto ys = ze * m.y();
      const auto zs = ze - xi;

      return {xs, ys, zs};
    }

    //! @brief Backproject a 2D pixel coordinates back to the corresponding 3D
    //! light ray.
    auto backproject(const Vec2& x) const -> Vec3
    {
      // Back to normalized camera coordinates.
      const Vec2 pd = (K_inverse.value() * x.homogeneous()).head(2);

      // Undistort the point.
      const Vec2 pu = undistort(pd);

      // Get the corresponding homogeneous equation.
      const Vec2 m = pu.homogeneous();

      // Apply the lifting transformation, that is the inverse of the mirror
      // transformation.
      const auto Xs = lifting(m);

      return Xs;
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
                          (mu + distortion_term(mu) - md).norm() > eps;
           ++iter)
      {
        mu = md - distortion_term(mu);
      }

      return mu;
    }
  };

}  // namespace DO::Sara
