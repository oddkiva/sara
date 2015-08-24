// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Geometry/Objects/Triangle.hpp>

namespace DO { namespace Sara {

  Triangle::Triangle(const Point2d& a, const Point2d& b, const Point2d& c)
  {
    _v[0] = a;
    _v[1] = b;
    _v[2] = c;
  }

} /* namespace Sara */
} /* namespace DO */
