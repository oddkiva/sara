// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018-2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/MultiViewGeometry/MinimalSolvers/Triangulation.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>


namespace DO::Sara {

  //! @addtogroup MultiViewGeometry
  //! @{

  struct TwoViewGeometry
  {
    PinholeCamera C1;
    PinholeCamera C2;
    Eigen::MatrixXd X;
    Eigen::VectorXd scales1;
    Eigen::VectorXd scales2;
    Eigen::Array<bool, 1, Eigen::Dynamic> cheirality;
  };

  inline auto two_view_geometry(const Motion& m, const MatrixXd& u1,
                                const MatrixXd& u2) -> TwoViewGeometry
  {
    const auto C1 = normalized_camera();
    const auto C2 = normalized_camera(m.R, m.t.normalized());
    const Matrix34d P1 = C1;
    const Matrix34d P2 = C2;
    const auto [X, s1, s2] = triangulate_linear_eigen(P1, P2, u1, u2);
    const auto cheirality = relative_motion_cheirality_predicate(X, P2);
    return {C1, C2, X, s1, s2, cheirality};
  }

  //! @}

} /* namespace DO::Sara */
