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

#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>


namespace DO::Sara {

auto cheirality_predicate(const MatrixXd& X) -> Array<bool, 1, Dynamic>
{
  if (X.rows() == 3)
    return X.row(2).array() > 0;

  if (X.rows() == 4)
    return X.colwise().hnormalized().row(2).array() > 0;

  throw std::runtime_error{"Error: invalid 3D coordinates!"};
}

auto relative_motion_cheirality_predicate(const MatrixXd& X, const Matrix34d& P)
    -> Array<bool, 1, Dynamic>
{
  return cheirality_predicate(X) && cheirality_predicate(P * X);
}

} /* namespace DO::Sara */
