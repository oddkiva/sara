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


namespace DO { namespace Sara {

  vector<Point2d> ramer_douglas_peucker(vector<Point2d> curve, double eps)
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
