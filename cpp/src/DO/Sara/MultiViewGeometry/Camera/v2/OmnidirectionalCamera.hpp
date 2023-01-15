#pragma once

#include <DO/Sara/MultiViewGeometry/Camera/v2/CameraIntrinsicBase.hpp>

#include <Eigen/Geometry>

#include <iostream>


namespace DO::Sara::v2 {

  template <ArrayConcept Array>
  struct OmnidirectionalCameraBase : CameraIntrinsicBase<Array>
  {
    using base_type = CameraIntrinsicBase<Array>;
    using base_type::data;
    using T = typename Array::value_type;

    // clang-format off
    enum Index
    {
      FX = 0,
      FY = 1,
      S  = 2,
      U0 = 3,
      V0 = 4,
      K0 = 5,
      K1 = 6,
      P0 = 7,
      P1 = 8,
      XI = 9
    };
    // clang-format on

    auto fx() -> T&
    {
      return data[FX];
    }

    auto fy() -> T&
    {
      return data[FY];
    }

    auto shear() -> T&
    {
      return data[S];
    }

    auto u0() -> T&
    {
      return data[U0];
    }

    auto v0() -> T&
    {
      return data[V0];
    }

    auto k() -> VectorView<T, 2>
    {
      return VectorView<T, 2>{data.data() + K0};
    }

    auto p() -> VectorView<T, 2>
    {
      return VectorView<T, 2>{data.data() + P0};
    }

    auto k(const int i) -> T&
    {
      return data[K0 + i];
    };

    auto p(const int i) -> T&
    {
      return data[P0 + i];
    };

    auto xi() -> T&
    {
      return data[XI];
    };

    auto fx() const -> T
    {
      return data[FX];
    }

    auto fy() const -> T
    {
      return data[FY];
    }

    auto shear() const -> T
    {
      return data[S];
    }

    auto u0() const -> T
    {
      return data[U0];
    }

    auto v0() const -> T
    {
      return data[V0];
    }

    auto k() const -> ConstVectorView<T, 2>
    {
      return ConstVectorView<T, 2>{data.data() + K0};
    }

    auto p() const -> ConstVectorView<T, 2>
    {
      return ConstVectorView<T, 2>{data.data() + P0};
    }

    auto k(const int i) const -> T
    {
      return data[K0 + i];
    };

    auto p(const int i) const -> T
    {
      return data[P0 + i];
    };

    auto xi() const -> T
    {
      return data[XI];
    };

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
    inline auto lifting(const Eigen::Vector2<T>& m) const -> Eigen::Vector3<T>
    {
      // Calculate auxiliary terms.
      const auto xi_squared = xi() * xi();
      const auto m_squared_norm = m.squaredNorm();

      const auto discriminant = 1 + (1 - xi_squared) * m_squared_norm;

      // Calculate the z-coordinate of Xe.
      const auto numerator = xi() + std::sqrt(discriminant);

      // Determine the z coordinate of the point on the sphere on the decentered
      // reference frame.
      const auto denominator = m_squared_norm + 1;
      const auto ze = numerator / denominator;

      // Back to the reference frame centered in the mirror centre.
      const auto xs = ze * m.x();
      const auto ys = ze * m.y();
      const auto zs = ze - xi();

      return {xs, ys, zs};
    }

    //! @brief Apply only in the normalized coordinates.
    auto lens_distortion(const Eigen::Vector2<T>& mu) const -> Eigen::Vector2<T>
    {
      // Distortion.
      const auto& k1 = k(0);
      const auto& k2 = k(1);
      const auto& p1 = p(0);
      const auto& p2 = p(1);

      // Radial component (additive).
      const auto r2 = mu.squaredNorm();
      const auto r4 = r2 * r2;
      const Eigen::Vector2<T> radial_factor = mu * (k1 * r2 + k2 * r4);

      // Tangential component (additive).
      static const auto two = static_cast<T>(2);
      const auto tx =
          two * p1 * mu.x() * mu.y() + p2 * (r2 + two * p1 * mu.x());
      const auto ty =
          p1 * (r2 + two * p1 * mu.y()) + two * p2 * mu.x() * mu.y();

      // Apply the distortion.
      const Eigen::Vector2<T> delta = radial_factor + Eigen::Vector2<T>{tx, ty};

      return delta;
    }

    //! @brief Iterative method to remove distortion.
    auto correct_lens_distortion(const Eigen::Vector2<T>& md,
                                 int num_iterations = 10) const
        -> Eigen::Vector2<T>
    {
      static constexpr auto eps = static_cast<T>(1e-8);

      auto mu = md;
      for (auto iter = 0; iter < num_iterations &&
                          (mu + lens_distortion(mu) - md).norm() > eps;
           ++iter)
        mu = md - lens_distortion(mu);

      return mu;
    }

    //! @brief Project a 3D scene point/ray expressed in the camera frame to the
    //! distorted image coordinates.
    auto project(const Eigen::Vector3<T>& Xc) const -> Eigen::Vector2<T>
    {
      // c) Calculate the 3D light ray.
      const Eigen::Vector3<T> Xs = Xc.normalized();
      const Eigen::Vector3<T> Xe = Xs + xi() * Eigen::Vector3<T>::UnitZ();
      const Eigen::Vector2<T> mu = Xe.hnormalized();

      // Lens distortion.
      const Eigen::Vector2<T> md = mu + lens_distortion(mu);

      // e) Project to the image plane.
      auto x = Eigen::Vector2<T>{};
      x.x() = fx() * md.x() + shear() * md.y() + u0();
      x.y() = fy() * md.y() + v0();

      return x;
    }

    //! @brief Backproject a 2D pixel coordinates back to the corresponding 3D
    //! light ray.
    inline auto backproject(const Eigen::Vector2<T>& uv) const
        -> Eigen::Vector3<T>
    {
      // Back to normalized camera coordinates.
      const auto& u = uv.x();
      const auto& v = uv.y();
      const auto y = (v - v0()) / fy();
      const auto x = (u - u0() - shear() * y) / fx();
      const auto pd = Eigen::Vector2<T>{x, y};

      // Undistort the point.
      const auto pu = correct_lens_distortion(pd);

      // Apply the lifting transformation, that is the inverse of the mirror
      // transformation.
      const auto Xs = lifting(pu);

      return Xs;
    }

    inline auto distort(const Eigen::Vector2<T>& xu) const -> Eigen::Vector2<T>
    {
      // Backproject the undistorted coordinates to a 3D ray.
      const auto& u = xu.x();
      const auto& v = xu.y();
      const auto y = (v - v0()) / fy();
      const auto x = (u - u0() - shear() * y) / fx();
      const auto xun = Eigen::Vector3<T>{x, y, 1};

      // Project the 3D ray to the normal image.
      const auto xd = project(xun);

      return xd;
    }

    inline auto undistort(const Eigen::Vector2<T>& x) const -> Eigen::Vector2<T>
    {
      const Eigen::Vector2<T> Xs = backproject(x).hnormalized();

      const auto u = fx() * Xs.x() + shear() * Xs.y() + u0();
      const auto v = fy() * Xs.y() + v0();

      return {u, v};
    }
  };

  template <typename T>
  using OmnidirectionalCameraView =
      OmnidirectionalCameraBase<VectorView<T, 10>>;

  template <typename T>
  using OmnidirectionalCameraConstView =
      OmnidirectionalCameraBase<ConstVectorView<T, 10>>;

  template <typename T>
  using OmnidirectionalCamera = OmnidirectionalCameraBase<Eigen::Vector<T, 10>>;

}  // namespace DO::Sara::v2
