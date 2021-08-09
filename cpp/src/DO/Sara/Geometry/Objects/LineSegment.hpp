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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  //! @addtogroup GeometryObjects
  //! @{

  class LineSegment : private std::pair<Point2d, Point2d>
  {
  public:
    using Base = std::pair<Point2d, Point2d>;

    LineSegment() = default;
    LineSegment(const Base& pair) noexcept : Base(pair) {}
    LineSegment(const Point2d& p1, const Point2d& p2) noexcept : Base(p1, p2) {}

    Point2d& p1() noexcept { return first; }
    Point2d& p2() noexcept { return second; }
    const Point2d& p1() const noexcept { return first; }
    const Point2d& p2() const noexcept { return second; }

    double x1() const noexcept { return p1().x(); }
    double y1() const noexcept { return p1().y(); }
    double& x1() noexcept { return p1().x(); }
    double& y1() noexcept { return p1().y(); }

    double x2() const noexcept { return p2().x(); }
    double y2() const noexcept { return p2().y(); }
    double& x2() noexcept { return p2().x(); }
    double& y2() noexcept { return p2().y(); }

    inline auto direction() const -> Eigen::Vector2d
    {
      return p2() - p1();
    }

    inline auto squared_length() const
    {
      return direction().squaredNorm();
    }

    inline auto length() const
    {
      return direction().norm();
    }
  };

  /*!
    Intersection test between line segments.
    'p' is the intersection point if it exists.
   */
  DO_SARA_EXPORT
  bool intersection(const LineSegment& s1, const LineSegment& s2, Point2d& p);

  //! @}

} /* namespace Sara */
} /* namespace DO */
