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
    using vector2_type = typename base_type::vector2_type;
    using vector3_type = typename base_type::vector3_type;

    //! @brief The distortion coefficients.
    vector3_type radial_distortion_coefficients;
    vector2_type tangential_distortion_coefficients;
    T xi;

    //! @brief Project a 3D scene point expressed in the camera frame to the
    //! distorted image coordinates.
    inline auto project(const vector3_type& X) const -> vector2_type
    {
      // Mirror transformation
      //
      // 1. Project on the unit sphere (reflection from the spherical mirror).
      const vector3_type Xs = X.normalized();
      // 2. Change coordinates.
      const vector3_type Xe = Xs + xi * vector3_type::UnitZ();
      // 3. Project the reflected ray by the mirror to the normalized plane z
      // = 1.
      const vector2_type m = Xe.hnormalized();

      // Apply the distortion.
      const vector2_type m_distorted = m + lens_distortion(m);

      // Go back to pixel coordinates.
      const vector2_type p = (K * m_distorted.homogeneous()).head(2);

      return p;
    }

    //! @brief Backproject a 2D pixel coordinates back to the corresponding 3D
    //! light ray.
    inline auto backproject(const vector2_type& x) const -> vector3_type
    {
      // Back to normalized camera coordinates.
      const vector2_type pd = (K_inverse * x.homogeneous()).head(2);

      // Undistort the point.
      const vector2_type pu = correct_lens_distortion(pd);

      // Apply the lifting transformation, that is the inverse of the mirror
      // transformation.
      const auto Xs = lifting(pu);

      return Xs;
    }

    inline auto undistort(const vector2_type& x) const -> vector2_type
    {
      const vector3_type Xs = backproject(x);
      const vector2_type xu = (K * Xs).hnormalized();
      return xu;
    }

    inline auto distort(const vector2_type& xu) const -> vector2_type
    {
      // Backproject the undistorted coordinates to a 3D ray.
      const vector3_type xun = K_inverse * xu.homogeneous();
      // Project the 3D ray to the normal image.
      const vector2_type xd = project(xun);
      return xd;
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
    inline auto lifting(const vector2_type& m) const -> vector3_type
    {
      // Calculate auxiliary terms.
      const auto xi_squared = xi * xi;
      const auto m_squared_norm = m.squaredNorm();

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

    //! @brief Apply only in the normalized coordinates.
    inline auto lens_distortion(const vector2_type& m_undistorted) const
        -> vector2_type
    {
      const auto& m = m_undistorted;
      using CameraModel32f = CameraModel<float>;
      // Distortion.
      const auto& k1 = radial_distortion_coefficients(0);
      const auto& k2 = radial_distortion_coefficients(1);
      const auto& p1 = tangential_distortion_coefficients(0);
      const auto& p2 = tangential_distortion_coefficients(1);

      // Radial component (additive).
      const auto r2 = m.squaredNorm();
      const auto r4 = r2 * r2;
      const vector2_type radial_factor = m * (k1 * r2 + k2 * r4);

      // Tangential component (additive).
      const auto tx = 2 * p1 * m.x() * m.y() + p2 * (r2 + 2 * p1 * m.x());
      const auto ty = p1 * (r2 + 2 * p1 * m.y()) + 2 * p2 * m.x() * m.y();

      // Apply the distortion.
      const vector2_type delta = radial_factor + vector2_type{tx, ty};

      return delta;
    }

    //! @brief Iterative method to remove distortion.
    inline auto correct_lens_distortion(const vector2_type& pd,
                                        int num_iterations = 10) const
        -> vector2_type
    {
      // Calculate the normalized coordinates.
      const vector2_type md = (K_inverse * pd.homogeneous()).head(2);

      auto mu = md;
      for (auto iter = 0; iter < num_iterations &&
                          (mu + lens_distortion(mu) - md).norm() > eps;
           ++iter)
        mu = md - lens_distortion(mu);

      return mu;
    }
  };

}  // namespace DO::Sara
