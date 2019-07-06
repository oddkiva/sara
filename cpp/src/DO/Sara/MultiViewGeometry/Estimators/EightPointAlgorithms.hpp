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

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  DO_SARA_EXPORT
  void eight_point_fundamental_matrix(const Matrix<double, 3, 8>& x,
                                      const Matrix<double, 3, 8>& y,
                                      Matrix3d& F);

  DO_SARA_EXPORT
  void four_point_homography(const Matrix<double, 3, 4>& x,
                             const Matrix<double, 3, 4>& y,  //
                             Matrix3d& H);

} /* namespace Sara */
} /* namespace DO */
