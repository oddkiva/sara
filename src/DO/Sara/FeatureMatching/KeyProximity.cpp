// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/FeatureMatching.hpp>


namespace DO { namespace Sara {

  bool KeyProximity::operator()(const OERegion& f1, const OERegion& f2) const
  {
    SquaredRefDistance<float, 2> m1{ mapped_squared_metric(f1) };
    SquaredRefDistance<float, 2> m2{ mapped_squared_metric(f2) };

    float sd1 = m1(f1.center(), f2.center());
    float sd2 = m2(f1.center(), f2.center());

    float squared_pixel_dist = (f1.center() - f2.center()).squaredNorm();

    return (squared_pixel_dist < _squared_dist_thres) ||
      (sd1 < _squared_metric_dist) || (sd2 < _squared_metric_dist);
  }

} /* namespace Sara */
} /* namespace DO */
