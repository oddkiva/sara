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

#ifndef DO_FEATUREMATCHING_MATCHFILTERING_HPP
#define DO_FEATUREMATCHING_MATCHFILTERING_HPP

namespace DO {

	class KeyProximity
	{
	public:
		KeyProximity(float metricDistT = .5f, float pixelDistT = 10.f)
			: sqMetricDist(metricDistT*metricDistT)
      , sqPixDist(pixelDistT*pixelDistT) {}

    SquaredRefDistance<float, 2> mappedSquaredMetric(const OERegion& f) const
    { return SquaredRefDistance<float, 2>(f.shapeMat()); }

		bool operator()(const Keypoint& k1, const Keypoint& k2) const;

  private:
    float sqMetricDist;
    float sqPixDist;
  };

} /* namespace DO */

#endif /* DO_FEATUREMATCHING_MATCHFILTERING_HPP */