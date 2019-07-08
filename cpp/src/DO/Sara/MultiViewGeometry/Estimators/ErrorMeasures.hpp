#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO::Sara {

//! @brief Functor evaluating distance of a point to its epipolar line.
struct EpipolarDistance
{
  EpipolarDistance() = default;

  EpipolarDistance(const Eigen::Matrix3d& F_)
    : F{F_}
  {
  }

  auto operator()(const Eigen::Vector3d& X, const Eigen::Vector3d& Y) const
  {
    return std::abs(Y.transpose() * F * X);
  };

  Eigen::Matrix3d F;
};


struct SymmetricTransferError
{
  SymmetricTransferError() = default;

  SymmetricTransferError(const Eigen::Matrix3d& H)
    : H_{H}
    , H_inv_{H.inverse()}
  {
  }

  auto operator()(const Eigen::Vector3d& x, const Eigen::Vector3d& y) const
      -> double
  {
    return ((H_ * x).hnormalized() - y.hnormalized()).norm() +
           ((H_inv_ * y).hnormalized() - x.hnormalized()).norm();
  };

  Eigen::Matrix3d H_;
  Eigen::Matrix3d H_inv_;
};


} /* namespace DO::Sara */
