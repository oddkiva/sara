// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Features/Feature.hpp>


namespace DO { namespace Sara {

	class DO_SARA_EXPORT AnnMatcher
	{
	public:
    //! @{
    //! @brief Constructors.
    AnnMatcher(const Set<OERegion, RealDescriptor>& keys1,
               const Set<OERegion, RealDescriptor>& keys2,
               float sift_ratio_thres = 1.2f);

    AnnMatcher(const Set<OERegion, RealDescriptor>& keys,
               float sift_ratio_thres = 1.2f,
               float min_max_metric_dist_thres = 0.5f,
               float pixel_dist_thres = 10.f);
    //! @}

    //! @{
    //! @brief Return matches.
    std::vector<Match> compute_matches();

		std::vector<Match> compute_self_matches()
    {
      return compute_matches();
    }
    //! @}

	private: /* data members */
    //! Input parameters.
    const Set<OERegion, RealDescriptor>& _keys1;
    const Set<OERegion, RealDescriptor>& _keys2;
		float _squared_ratio_thres;
    //! Internals.
    KeyProximity _is_too_close;
    std::size_t _max_neighbors;
    std::vector<int> _vec_indices;
    std::vector<float> _vec_dists;
    bool _self_matching;
	};

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATUREMATCHING_ANNMATCHER_HPP */
