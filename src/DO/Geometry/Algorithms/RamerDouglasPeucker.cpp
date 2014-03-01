// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_GEOMETRY_ALGORITHMS_RAMERDOUGLASPEUCKER_HPP
#define DO_GEOMETRY_ALGORITHMS_RAMERDOUGLASPEUCKER_HPP

#include <DO/Core/EigenExtension.hpp>
#include <DO/Core/StdVectorHelpers.hpp>

namespace DO {

  namespace internal {

    static
    inline
    double squaredDistance(const Point2d& a, const Point2d& b, const Point2d& x)
    {
      Matrix2d M;
      M.col(0) = b-a;
      M.col(1) = x-a;
      return std::abs(M.determinant());
    }

    static
    void
    ramerDouglasPeucker(std::vector<Point2d>& lines,
                        const std::vector<Point2d>& contours,
                        std::size_t begin, std::size_t end,
                        double eps)
    {
      if (end-begin < 3)
        return;

      if (lines.empty() || lines.back() != contours[begin])
        lines.push_back(contours[begin]);

      std::size_t index = begin+1;
      double maxDist = 0;
      for (std::size_t i = begin+1; i != end-1; ++i)
      {
        double dist = squaredDistance(contours[begin], contours[end-1],
                                      contours[i]);
        if (maxDist < dist)
        {
          index = i;
          maxDist = dist;
        }
      }

      if (maxDist > eps)
      {
        ramerDouglasPeucker(lines, contours, begin, index+1, eps);
        ramerDouglasPeucker(lines, contours, index, end, eps);
      }

      lines.push_back(contours[end-1]);
    }

  } /* namespace internal */

  std::vector<Point2d>
  ramerDouglasPeucker(const std::vector<Point2d>& contours, double eps)
  {
    std::vector<Point2d> lines;
    lines.reserve(lines.size());
    internal::ramerDouglasPeucker(lines, contours, 0, contours.size(), eps);
    shrink_to_fit(lines);
    return lines;
  }
}

#endif /* DO_GEOMETRY_ALGORITHMS_RAMERDOUGLASPEUCKER_HPP */