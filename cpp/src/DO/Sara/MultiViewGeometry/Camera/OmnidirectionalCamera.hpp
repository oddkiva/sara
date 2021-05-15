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

    //! Iterative method to remove distortion.
    auto undistort(const Vec2& md, int num_iterations = 10) const -> Vec2
    {
      // Calculate the normalized coordinates.
      auto mu = md;
      for (auto iter = 0; iter < num_iterations &&
                          (mu + distortion_term(mu) - md).norm() > eps;
           ++iter)
        mu = md - distortion_term(mu);

      // const auto& k = radial_distortion_coefficients;
      // const auto& p = tangential_distortion_coefficients;
      // for (auto iter = 0; iter < 20; ++iter)
      // {
      //   const auto r2 = mu.squaredNorm();
      //   const auto r4 = r2 * r2;

      //   const auto denominator = 1 + k(0) * r2 + k(1) * r4;
      //   mu(0) = (md(0) - 2 * p(0) * mu(0) * mu(1) -
      //            p(1) * (r2 + 2 * mu(0) * mu(0))) /
      //           denominator;
      //   mu(1) = (md(1) - 2 * p(1) * mu(0) * mu(1) -
      //            p(0) * (r2 + 2 * mu(1) * mu(1))) /
      //           denominator;

      //   std::cout << "mu[" << iter << "] = " << mu.transpose() << std::endl;
      // }

      return mu;
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
     *  in z by exploiting one key algebraic observation:
     *  (2) xs^2 + ys^2 + zs^2 = 1 because the world points are reflected the
     *                             spherical mirror of radius = 1.
     *
     *  IMPORTANT:
     *  After observing carefully Figure 4 in the paper:
     *  https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
     *  is that by design the lower hemisphere of the mirror can project world
     *  points behind the omnidirectional camera and if the camera film plane is
     *  wide enough.
     *
     */
    auto lifting(const Vec2& m) const -> Vec3
    {
      // Calculate auxiliary terms.
      const auto xi_squared = xi * xi;
      const auto m_squared_norm = m.squaredNorm();

      // In the paper: https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
      // Because xi < 1, the discriminant is > 1.
      const auto discriminant = 1 + (1 - xi_squared) * m_squared_norm;

      // Calculate the z-coordinate of Xe.
      const auto numerator = xi + std::sqrt(discriminant);

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
      if (!K_inverse.has_value())
        base_type::cache_inverse_calibration_matrix();
      const Vec2 pd = (K_inverse.value() * x.homogeneous()).hnormalized();

      // Undistort the point.
      const Vec2 pu = undistort(pd);

      // Apply the lifting transformation, that is the inverse of the mirror
      // transformation.
      const auto Xs = lifting(pu);

      return Xs;
    }

    auto undistort_v2(const Vec2& x) const -> Vec2
    {
      const Vec3 Xs = backproject(x);
      // We can tell whether the point is behind the camera.
      const Vec2 xu = (K * Xs).hnormalized();
      return xu;
    }

    auto distort_v2(const Vec2 &xu) const -> Vec2
    {
      // Normalized camera coordinates.
      const Vec3 xun = (K_inverse.value() * xu.homogeneous());
      // Project the 3D ray to the normal image.
      const Vec2 xd = project(xun);
      return xd;
    }
  };

}  // namespace DO::Sara
