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

#pragma warning (disable : 4267 4503)

#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>
#include <iostream>
#include <fstream>

using namespace std;

namespace DO {

  Quad::Quad(const BBox& bbox)
  {
    v_[0] = bbox.topLeft();
    v_[1] = bbox.topRight();
    v_[2] = bbox.bottomRight();
    v_[3] = bbox.bottomLeft();
  }

  Quad::Quad(const Point2d& a, const Point2d& b,
             const Point2d& c, const Point2d& d)
  {
    v_[0] = a;
    v_[1] = b;
    v_[2] = c;
    v_[3] = d;
  }

} /* namespace DO */