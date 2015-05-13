// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Geometry/Objects/Cone.hpp>
#include <DO/Sara/Geometry/Tools/Utilities.hpp>

namespace DO {

  template class Cone<2>;
  template class Cone<3>;
  template class AffineCone<2>;
  template class AffineCone<3>;

  AffineCone2 affine_cone2(double theta0, double theta1, const Point2d& vertex)
  {
    Point2d u0, u1;
    u0 = unit_vector2(theta0);
    u1 = unit_vector2(theta1);
    return AffineCone2(u0, u1, vertex, AffineCone2::Convex);
  }

} /* namespace DO */