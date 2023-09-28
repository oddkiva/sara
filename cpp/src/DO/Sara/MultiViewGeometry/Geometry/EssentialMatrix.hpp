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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>


namespace DO::Sara {

  //! @ingroup MultiViewGeometry
  //! @defgroup EssentialMatrix Essential Matrix
  //! @{

  //! @brief Relative Pose
  //! @{
  struct Motion
  {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
  };

  class EssentialMatrix : public FundamentalMatrix
  {
    using base_type = FundamentalMatrix;

  public:
    using matrix_type = typename base_type::matrix_type;
    using point_type = typename base_type::point_type;
    using vector_type = point_type;

    EssentialMatrix() = default;

    EssentialMatrix(const matrix_type& E)
      : base_type{E}
    {
    }
  };

  inline auto essential_matrix(const Matrix3d& R, const Vector3d& t)
  {
    return EssentialMatrix{skew_symmetric_matrix(t) * R};
  }

  inline auto essential_matrix(const Motion& m)
  {
    return essential_matrix(m.R, m.t);
  }

  DO_SARA_EXPORT
  auto extract_relative_motion_svd(const Matrix3d& E) -> std::vector<Motion>;

  DO_SARA_EXPORT
  auto extract_relative_motion_horn(const Matrix3d& E) -> std::vector<Motion>;
  //! @}

  //! @}

} /* namespace DO::Sara */
