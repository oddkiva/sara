// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  class LineSegment : private std::pair<Point2d, Point2d>
  {
  public:
    using Base = std::pair<Point2d, Point2d>;
    LineSegment() {}
    LineSegment(const Base& pair) : Base(pair) {}
    LineSegment(const Point2d& p1, const Point2d& p2) : Base(p1, p2) {}

    Point2d& p1() { return first; }
    Point2d& p2() { return second; }
    const Point2d& p1() const { return first; }
    const Point2d& p2() const { return second; }

    double x1() const { return p1().x(); }
    double y1() const { return p1().y(); }
    double& x1() { return p1().x(); }
    double& y1() { return p1().y(); }

    double x2() const { return p2().x(); }
    double y2() const { return p2().y(); }
    double& x2() { return p2().x(); }
    double& y2() { return p2().y(); }
  };

  inline Vector2d dir(const LineSegment& s)
  {
    return s.p2() - s.p1();
  }

  inline double squared_length(const LineSegment& s)
  {
    return (s.p2()-s.p1()).squaredNorm();
  }

  inline double length(const LineSegment& s)
  {
    return (s.p2()-s.p1()).norm();
  }

  /*!
    Intersection test between line segments.
    'p' is the intersection point if it exists.
   */
  bool intersection(const LineSegment& s1, const LineSegment& s2, Point2d& p);

} /* namespace Sara */
} /* namespace DO */
