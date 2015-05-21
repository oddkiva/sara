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

#include <DO/Sara/FeatureMatching.hpp>

namespace DO {

  bool KeyProximity::operator()(const OERegion& f1, const OERegion& f2) const
  {
    SquaredRefDistance<float, 2> m1(mappedSquaredMetric(f1));
    SquaredRefDistance<float, 2> m2(mappedSquaredMetric(f1));

    float sd1 = m1(f1.center(), f2.center());
    float sd2 = m2(f1.center(), f2.center());

    float pixelDist2 = (f1.center() - f2.center()).squaredNorm();

    return (pixelDist2 < sqPixDist) ||
      (sd1 < sqMetricDist) || (sd2 < sqMetricDist);
  }

} /* namespace DO */