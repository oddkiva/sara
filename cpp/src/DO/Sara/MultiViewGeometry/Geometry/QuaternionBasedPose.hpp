#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>


namespace DO::Sara {

  template <typename T>
  struct QuaternionBasedPose
  {
    Eigen::Quaternion<T> q;
    Eigen::Vector3<T> t;

    inline auto operator*(const Eigen::Vector3<T>& x) const -> Eigen::Vector3<T>
    {
      return q * x + t;
    }

    inline auto matrix34() const -> Eigen::Matrix<T, 3, 4>
    {
      return (Eigen::Matrix<T, 3, 4>{} << q.toRotationMatrix(), t).finished();
    }

    inline auto matrix4() const -> Eigen::Matrix4<T>
    {
      auto r = Eigen::Matrix<T, 4, 4>{};
      r << matrix34(), Eigen::RowVector3<T>::Zero().homogeneous();
      return r;
    }

    static inline auto identity() -> QuaternionBasedPose<T>
    {
      return {.q = Eigen::Quaternion<T>::Identity(),
              .t = Eigen::Vector3<T>::Zero()};
    }
  };

}  // namespace DO::Sara
