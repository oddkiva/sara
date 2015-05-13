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


#ifndef DO_SARA_FEATUREMATCHING_ANNMATCHER_HPP
#define DO_SARA_FEATUREMATCHING_ANNMATCHER_HPP

namespace DO
{
	class AnnMatcher
	{
	public:
    //! Constructors
    AnnMatcher(const Set<OERegion, RealDescriptor>& keys1,
               const Set<OERegion, RealDescriptor>& keys2,
               float siftRatioT = 1.2f);
    AnnMatcher(const Set<OERegion, RealDescriptor>& keys,
               float siftRatioT = 1.2f,
               float minMaxMetricDistT = 0.5f,
               float pixelDistT = 10.f);

    std::vector<Match> computeMatches();
		std::vector<Match> computeSelfMatches() { return computeMatches(); }

	private: /* data members */
    //! Input parameters
    const Set<OERegion, RealDescriptor>& keys1_;
    const Set<OERegion, RealDescriptor>& keys2_;
		float sqRatioT;
    //! Internals
    KeyProximity is_too_close_;
    std::size_t max_neighbors_;
    std::vector<int> vec_indices_;
    std::vector<float> vec_dists_;
    bool self_matching_;
	};

} /* namespace DO */

#endif /* DO_SARA_FEATUREMATCHING_ANNMATCHER_HPP */