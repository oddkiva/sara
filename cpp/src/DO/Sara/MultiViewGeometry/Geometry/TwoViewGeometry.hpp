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
#include <DO/Sara/MultiViewGeometry/Estimators/Triangulation.hpp>


namespace DO::Sara {

struct TwoViewGeometry
{
  PinholeCamera C1;
  PinholeCamera C2;
  MatrixXd X;
};

inline auto two_view_geometry(const Motion& m, const MatrixXd& x1,
                              const MatrixXd& x2) -> TwoViewGeometry
{
  const auto C1 = normalized_camera();
  const auto C2 = normalized_camera(m.R, m.t.normalized());
  const Matrix34d P1 = C1;
  const Matrix34d P2 = C2;
  const auto X = triangulate_linear_eigen(P1, P2, x1, x2);
  return {C1, C2, X};
}

inline auto remove_cheirality_inconsistent_geometries(
    std::vector<TwoViewGeometry>& geometries)
{
  geometries.erase(std::remove_if(std::begin(geometries), std::end(geometries),
                                  [&](const TwoViewGeometry& g) {
                                    return !cheirality_predicate(g.X, g.C1) ||
                                           !cheirality_predicate(g.X, g.C2);
                                  }),
                   std::end(geometries));
}

} /* namespace DO::Sara */
