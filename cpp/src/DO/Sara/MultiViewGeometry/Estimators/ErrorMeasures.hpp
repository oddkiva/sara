#pragma once

#include <Eigen/Core>


namespace DO::Sara {

//! @brief Functor evaluating distance of a point to its epipolar line.
struct EpipolarDistance
{
  EpipolarDistance(const Eigen::Matrix3d& F_)
    : F{F_}
  {
  }

  inline auto operator()(const Eigen::Vector3d& X,
                         const Eigen::Vector3d& Y) const
  {
    return std::abs(Y.transpose() * F * X);
  };

  const Eigen::Matrix3d& F;
};


struct SymmetricTransferError
{
  inline SymmetricTransferError(Eigen::Matrix3d& H)
    : H_{H}
    , H_inv_{H.inverse()}
  {
  }

  inline auto operator()(const Eigen::Matrix3d& H,  //
                         const Eigen::Vector3d& x,
                         const Eigen::Vector3d& y) const -> double
  {
    return ((H_ * x).hnormalized() - y.hnormalized()).norm() +
           ((H_inv_ * y).hnormalized() - x.hnormalized()).norm();
  };

  Eigen::Matrix3d H_;
  Eigen::Matrix3d H_inv_;
};


} /* namespace DO::Sara */
