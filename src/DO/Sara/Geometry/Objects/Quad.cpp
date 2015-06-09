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

#include <DO/Sara/Geometry/Objects/Quad.hpp>

using namespace std;

namespace DO { namespace Sara {

  Quad::Quad(const BBox& bbox)
  {
    v_[0] = bbox.top_left();
    v_[1] = bbox.top_right();
    v_[2] = bbox.bottom_right();
    v_[3] = bbox.bottom_left();
  }

  Quad::Quad(const Point2d& a, const Point2d& b,
             const Point2d& c, const Point2d& d)
  {
    v_[0] = a;
    v_[1] = b;
    v_[2] = c;
    v_[3] = d;
  }

} /* namespace Sara */
} /* namespace DO */
