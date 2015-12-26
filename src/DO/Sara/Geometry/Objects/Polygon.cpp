// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Geometry/Objects/Polygon.hpp>


namespace DO { namespace Sara {

  double area(const std::vector<Point2d>& polygon)
  {
    //! Computation derived from Green's formula
    double A = 0.;
    int N = int(polygon.size());
    for (int i1 = N-1, i2 = 0; i2 < N; i1=i2++)
    {
      Matrix2d M;
      M.col(0) = polygon[i1];
      M.col(1) = polygon[i2];
      A += M.determinant();
    }
    return fabs(0.5*A);
  }

} /* namespace Sara */
} /* namespace DO */
