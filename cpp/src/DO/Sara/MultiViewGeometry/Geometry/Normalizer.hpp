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

#include <DO/Sara/Geometry/Tools/Normalizer.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/Homography.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>
#include <DO/Sara/MultiViewGeometry/PointCorrespondenceList.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>


namespace DO::Sara {

  //! @ingroup GeometryDataNormalizer

  //! @{

  //! @brief Normalizer for the two-view homography estimation.
  template <>
  struct Normalizer<Homography>
  {
    Normalizer(const TensorView_<double, 2>& p1,
               const TensorView_<double, 2>& p2)
      : T1{compute_normalizer(p1)}
      , T2{compute_normalizer(p2)}
    {
      T1_inv = T1.inverse();
      T2_inv = T2.inverse();
    }

    Normalizer(const PointCorrespondenceList<double>& matches)
      : T1{compute_normalizer(matches.x)}
      , T2{compute_normalizer(matches.y)}
    {
      T1_inv = T1.inverse();
      T2_inv = T2.inverse();
    }

    auto normalize(const TensorView_<double, 2>& p1,
                   const TensorView_<double, 2>& p2) const
    {
      return std::make_tuple(apply_transform(T1, p1), apply_transform(T2, p2));
    }

    auto normalize(const PointCorrespondenceList<double>& M) const
        -> PointCorrespondenceList<double>
    {
      auto Mn = PointCorrespondenceList<double>{};
      std::tie(Mn.x, Mn.y) = this->normalize(M.x, M.y);
      return Mn;
    }

    inline auto denormalize(Eigen::Matrix3d& H) const -> void
    {
      H = T2_inv * H * T1;
    }

    Eigen::Matrix3d T1;
    Eigen::Matrix3d T2;
    Eigen::Matrix3d T1_inv;
    Eigen::Matrix3d T2_inv;
  };

  //! @brief Normalizer for the two-view fundamental matrix estimation.
  template <>
  struct Normalizer<FundamentalMatrix>
  {
    Normalizer(const TensorView_<double, 2>& p1,
               const TensorView_<double, 2>& p2)
      : T1{compute_normalizer(p1)}
      , T2{compute_normalizer(p2)}
    {
    }

    Normalizer(const PointCorrespondenceList<double>& M)
      : T1{compute_normalizer(M.x)}
      , T2{compute_normalizer(M.y)}
    {
    }

    auto normalize(const TensorView_<double, 2>& p1,
                   const TensorView_<double, 2>& p2) const
    {
      return std::make_tuple(apply_transform(T1, p1), apply_transform(T2, p2));
    }

    auto normalize(const PointCorrespondenceList<double>& M) const
        -> PointCorrespondenceList<double>
    {
      auto Mn = PointCorrespondenceList<double>{};
      std::tie(Mn.x, Mn.y) = this->normalize(M.x, M.y);
      return Mn;
    }

    auto denormalize(Eigen::Matrix3d& F) const -> void
    {
      F = (T2.transpose() * F * T1).normalized();
    }

    Eigen::Matrix3d T1;
    Eigen::Matrix3d T2;
  };

  //! @brief Only valid for ideal pinhole cameras.
  template <>
  struct Normalizer<EssentialMatrix>
  {
    Normalizer(const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2)
      : K1_inv{K1.inverse()}
      , K2_inv{K2.inverse()}
    {
    }

    auto normalize(const TensorView_<double, 2>& p1,
                   const TensorView_<double, 2>& p2) const
    {
      auto p1n = apply_transform(K1_inv, p1);
      auto p2n = apply_transform(K2_inv, p2);

      // Normalize backprojected rays to unit norm.
      p1n.colmajor_view().matrix().colwise().normalize();
      p2n.colmajor_view().matrix().colwise().normalize();

      return std::make_tuple(p1n, p2n);
    }

    auto normalize(const PointCorrespondenceList<double>& M) const
        -> PointCorrespondenceList<double>
    {
      auto Mn = PointCorrespondenceList<double>{};
      std::tie(Mn.x, Mn.y) = this->normalize(M.x, M.y);
      return Mn;
    }

    //! @brief Dummy implementation.
    auto denormalize(const Eigen::Matrix3d&) const -> void
    {
    }

    Eigen::Matrix3d K1_inv;
    Eigen::Matrix3d K2_inv;
  };

  //! @brief Normalizer for the two-view relative pose estimation.
  template <>
  struct Normalizer<TwoViewGeometry>
  {
    Normalizer(const Eigen::Matrix3d& K1, const Eigen::Matrix3d& K2)
      : K1_inv{K1.inverse()}
      , K2_inv{K2.inverse()}
    {
    }

    auto normalize(const TensorView_<double, 2>& p1,
                   const TensorView_<double, 2>& p2) const
    {
      auto p1n = apply_transform(K1_inv, p1);
      auto p2n = apply_transform(K2_inv, p2);

      // Normalize backprojected rays to unit norm.
      p1n.colmajor_view().matrix().colwise().normalize();
      p2n.colmajor_view().matrix().colwise().normalize();

      return std::make_tuple(p1n, p2n);
    }

    auto normalize(const PointCorrespondenceList<double>& M) const
        -> PointCorrespondenceList<double>
    {
      auto Mn = PointCorrespondenceList<double>{};
      std::tie(Mn.x, Mn.y) = this->normalize(M.x, M.y);
      return Mn;
    }

    //! @brief Dummy implementation.
    auto denormalize(const TwoViewGeometry&) const -> void
    {
    }

    Eigen::Matrix3d K1_inv;
    Eigen::Matrix3d K2_inv;
  };

  //! @brief Normalizer for the PnP estimation.
  template <>
  struct Normalizer<Eigen::Matrix<double, 3, 4>>
  {
    using PoseMatrix = Eigen::Matrix<double, 3, 4>;

    Normalizer() = default;

    auto normalize(const TensorView_<double, 2>& scene_points,
                   const TensorView_<double, 2>& backprojected_rays) const
    {
      return std::make_tuple(scene_points, backprojected_rays);
    }

    auto normalize(const PointCorrespondenceList<double>& M) const
        -> PointCorrespondenceList<double>
    {
      return M;
    }

    auto denormalize(const PoseMatrix&) const -> void
    {
    }
  };

  //! @}

} /* namespace DO::Sara */
