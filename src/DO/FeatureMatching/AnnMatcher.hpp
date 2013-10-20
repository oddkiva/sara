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


#ifndef DO_FEATUREMATCHING_ANNMATCHER_HPP
#define DO_FEATUREMATCHING_ANNMATCHER_HPP

namespace DO
{
	class AnnMatcher
	{
	public:
    //! Constructors
    AnnMatcher(const std::vector<Keypoint>& keys1,
               const std::vector<Keypoint>& keys2,
               float siftRatioT = 1.2f);

    AnnMatcher(const std::vector<Keypoint>& keys, float siftRatioT = 1.2f,
               float minMaxMetricDistT = 0.5f, float pixelDistT = 10.f);

    std::vector<Match> computeMatches();
    std::vector<Match> computeSelfMatches() { return computeMatches(); }
		
	private: /* data members */
		//! Input parameters
    const std::vector<Keypoint>& keys1_;
    const std::vector<Keypoint>& keys2_;
		float sqRatioT;

		//! Internals
		KeyProximity is_too_close_;
		std::size_t max_neighbors_;
		std::vector<int> vec_indices_;
		std::vector<float> vec_dists_;
		bool self_matching_;
	};

} /* namespace DO */

#endif /* DO_FEATUREMATCHING_ANNMATCHER_HPP */