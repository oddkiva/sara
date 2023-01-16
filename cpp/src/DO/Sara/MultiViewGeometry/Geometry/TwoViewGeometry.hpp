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

#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/Triangulation.hpp>

#include <iostream>


namespace DO::Sara {

  //! @addtogroup MultiViewGeometry
  //! @{

  struct TwoViewGeometry
  {
    BasicPinholeCamera C1;
    BasicPinholeCamera C2;
    Eigen::MatrixXd X;
    Eigen::VectorXd scales1;
    Eigen::VectorXd scales2;
    Eigen::Array<bool, 1, Eigen::Dynamic> cheirality;

    friend auto operator<<(std::ostream& os, const TwoViewGeometry& g) -> std::ostream&
    {
      os << "Camera matrices\n";
      os << "C[1] =\n" << g.C1.matrix() << "\n";
      os << "C[2] =\n" << g.C2.matrix() << "\n";

      os << "Triangulated points\n";
      const Eigen::MatrixXd C1X = g.C1.matrix() * g.X;
      const Eigen::MatrixXd C2X = g.C2.matrix() * g.X;
      os << "X[1] =\n" << C1X << "\n";
      os << "X[2] =\n" << C2X << "\n";

      os << "Triangulated points\n";
      os << "scales[1] =\n" << g.scales1.transpose() << "\n";
      os << "scales[2] =\n" << g.scales2.transpose() << "\n";

      os << "Cheirality " << g.cheirality << "\n";

      return os;
    }
  };

  inline auto two_view_geometry(const Motion& m, const MatrixXd& u1,
                                const MatrixXd& u2) -> TwoViewGeometry
  {
    const auto C1 = normalized_camera();
    const auto C2 = normalized_camera(m.R, m.t.normalized());
    const Matrix34d P1 = C1;
    const Matrix34d P2 = C2;
    const auto [X, s1, s2] = triangulate_linear_eigen(P1, P2, u1, u2);
    const Eigen::Array<bool, 1, Eigen::Dynamic> cheirality =
        relative_motion_cheirality_predicate(X, P2);
    return {C1, C2, X, s1, s2, cheirality};
  }

  //! @}

} /* namespace DO::Sara */
