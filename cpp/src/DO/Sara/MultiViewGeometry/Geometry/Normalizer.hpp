// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/Homography.hpp>
#include <DO/Sara/MultiViewGeometry/PointCorrespondenceList.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>


namespace DO::Sara {

  //! @ingroup MultiViewGeometry
  //! @defgroup MultiviewNormalizer Normalizers
  //! @{

  template <typename Model>
  struct Normalizer
  {
  };


  template <>
  struct Normalizer<Homography>
  {
    inline Normalizer(const TensorView_<double, 2>& p1,
                      const TensorView_<double, 2>& p2)
      : T1{compute_normalizer(p1)}
      , T2{compute_normalizer(p2)}
    {
      T1_inv = T1.inverse();
      T2_inv = T2.inverse();
    }

    inline Normalizer(const PointCorrespondenceList<double>& matches)
      : T1{compute_normalizer(matches._p1)}
      , T2{compute_normalizer(matches._p2)}
    {
      T1_inv = T1.inverse();
      T2_inv = T2.inverse();
    }


    inline auto normalize(const TensorView_<double, 2>& p1,
                          const TensorView_<double, 2>& p2) const
    {
      return std::make_tuple(apply_transform(T1, p1), apply_transform(T2, p2));
    }

    inline auto normalize(const PointCorrespondenceList<double>& X) const
        -> PointCorrespondenceList<double>
    {
      auto Xn = PointCorrespondenceList<double>{};
      std::tie(Xn._p1, Xn._p2) = this->normalize(X._p1, X._p2);
      return Xn;
    }

    inline auto denormalize(const Eigen::Matrix3d& H) const -> Eigen::Matrix3d
    {
      return T2_inv * H * T1;
    }

    Eigen::Matrix3d T1;
    Eigen::Matrix3d T2;
    Eigen::Matrix3d T1_inv;
    Eigen::Matrix3d T2_inv;
  };


  template <>
  struct Normalizer<FundamentalMatrix>
  {
    inline Normalizer(const TensorView_<double, 2>& p1,
                      const TensorView_<double, 2>& p2)
      : T1{compute_normalizer(p1)}
      , T2{compute_normalizer(p2)}
    {
    }

    inline auto normalize(const TensorView_<double, 2>& p1,
                          const TensorView_<double, 2>& p2) const
    {
      return std::make_tuple(apply_transform(T1, p1), apply_transform(T2, p2));
    }

    inline auto denormalize(const Eigen::Matrix3d& F) const -> Eigen::Matrix3d
    {
      return (T2.transpose() * F * T1).normalized();
    }

    Eigen::Matrix3d T1;
    Eigen::Matrix3d T2;
  };


  template <>
  struct Normalizer<EssentialMatrix>
  {
    inline Normalizer(const TensorView_<double, 2>& p1,
                      const TensorView_<double, 2>& p2)
      : T1{compute_normalizer(p1)}
      , T2{compute_normalizer(p2)}
    {
    }

    inline auto normalize(const TensorView_<double, 2>& p1,
                          const TensorView_<double, 2>& p2) const
    {
      return std::make_tuple(apply_transform(T1, p1), apply_transform(T2, p2));
    }

    inline auto denormalize(const Eigen::Matrix3d& E) const -> Eigen::Matrix3d
    {
      return (T2.transpose() * E * T1).normalized();
    }

    Eigen::Matrix3d T1;
    Eigen::Matrix3d T2;
  };

  //! @}

} /* namespace DO::Sara */
