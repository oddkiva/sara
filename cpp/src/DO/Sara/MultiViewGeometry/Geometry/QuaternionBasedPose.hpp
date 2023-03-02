#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>


namespace DO::Sara {

  template <typename T>
  struct QuaternionBasedPose
  {
    Eigen::Quaternion<T> q;
    Eigen::Vector3<T> t;

    inline auto operator()(const Eigen::Vector3<T>& x) const
        -> Eigen::Vector3<T>
    {
      return q * x + t;
    }

    inline auto matrix34() const -> Eigen::Matrix<T, 3, 4>
    {
      return (Eigen::Matrix<T, 3, 4>{} << q.toRotationMatrix(), t).finished();
    }

    inline auto matrix4() const -> Eigen::Matrix4<T>
    {
      return (Eigen::Matrix<T, 4, 4>{} <<
              matrix34(), Eigen::RowVector3<T>::Zero().homogeneous())
          .finished();
    }
  };

}  // namespace DO::Sara
