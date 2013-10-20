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

#include <DO/FeatureMatching.hpp>

namespace DO {

  bool KeyProximity::operator()(const Keypoint& k1, const Keypoint& k2) const
  {
    SquaredRefDistance<float, 2> m1(mappedSquaredMetric(k1.feat()));
    SquaredRefDistance<float, 2> m2(mappedSquaredMetric(k2.feat()));

    float sd1 = m1(k1.feat().center(), k2.feat().center());
    float sd2 = m2(k1.feat().center(), k2.feat().center());

    float pixelDist2 = (k1.feat().center() - k2.feat().center()).squaredNorm();

    return (pixelDist2 < sqPixDist) || 
      (sd1 < sqMetricDist) || (sd2 < sqMetricDist);
  }

} /* namespace DO */