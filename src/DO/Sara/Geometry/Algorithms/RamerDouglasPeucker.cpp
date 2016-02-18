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

#include <DO/Sara/Geometry.hpp>


using namespace std;


namespace DO { namespace Sara { namespace detail {

  double orthogonal_distance(const Point2d& a, const Point2d& b,
                             const Point2d& x)
  {
    auto M = Matrix2d{};
    M.col(0) = (b - a).normalized();
    M.col(1) = x - a;
    return abs(M.determinant());
  }

  vector<Point2d> ramer_douglas_peucker(const Point2d *in_first, const Point2d *in_last,
                                        double eps)
  {
    if (in_first == in_last)
      return { *in_first };

    auto pivot = in_first;
    auto pivot_dist = 0.;

    for (auto p = in_first + 1; p != in_last + 1; ++p)
    {
      auto dist = orthogonal_distance(*in_first, *in_last, *p);
      if (pivot_dist < dist)
      {
        pivot = p;
        pivot_dist = dist;
      }
    }

    auto out = vector<Point2d>{};
    if (pivot_dist > eps)
    {
      auto v1 = ramer_douglas_peucker(in_first, pivot, eps);
      auto v2 = ramer_douglas_peucker(pivot, in_last, eps);

      out.insert(out.end(), v1.begin(), v1.end());
      if (!v2.empty())
        out.insert(out.end(), v2.begin() + 1, v2.end());
    }
    else
      out = { *in_first, *in_last };

    return out;
  }

} /* namespace detail */
} /* namespace Sara */
} /* namespace DO */


namespace DO { namespace Sara {

  vector<Point2d>
  ramer_douglas_peucker(vector<Point2d> curve, double eps)
  {
    if (curve.empty())
      return {};

    // Remove consecutive coincident points.
    auto coincident = [](const Point2d& a, const Point2d& b) {
      return (a - b).squaredNorm() < 1e-8;
    };

    const auto it = unique(curve.begin(), curve.end(), coincident);
    curve.resize(it - curve.begin());

    if (curve.size() > 1 && coincident(curve.front(), curve.back()))
      curve.pop_back();

    // Run the algorithm on the cleaned curve.
    return detail::ramer_douglas_peucker(&curve.front(), &curve.back(), eps);
  }

} /* namespace Sara */
} /* namespace DO */