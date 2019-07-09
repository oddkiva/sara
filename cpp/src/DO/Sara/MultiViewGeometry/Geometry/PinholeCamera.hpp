#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO::Sara {

struct PinholeCamera
{
  operator Matrix34d() const
  {
    Matrix34d Rt = Matrix34d::Zero();
    Rt.topLeftCorner(3, 3) = R;
    Rt.col(3) = t;
    return K * Rt;
  }

  Eigen::Matrix3d K;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
};

} /* namespace DO::Sara */
