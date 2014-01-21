// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>

namespace DO {

  Triangle::Triangle(const Point2d& a, const Point2d& b, const Point2d& c)
  {
    v_[0] = a;
    v_[1] = b;
    v_[2] = c;
  }

}