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

//! @file

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/FeatureMatching/KeyProximity.hpp>
#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/Match/Match.hpp>


namespace DO { namespace Sara {

  /*!
   *  @addtogroup FeatureMatching
   *  @{
   */

  //! @brief Feature matcher class using FLANN's KD-Tree.
  class AnnMatcher
  {
  public:
    //! @brief Constructors.
    //! @{
    DO_SARA_EXPORT
    AnnMatcher(const KeypointList<OERegion, float>& keys1,
               const KeypointList<OERegion, float>& keys2,
               float sift_ratio_thres = 1.2f);

    DO_SARA_EXPORT
    AnnMatcher(const KeypointList<OERegion, float>& keys,
               float sift_ratio_thres = 1.2f,
               float min_max_metric_dist_thres = 0.5f,
               float pixel_dist_thres = 10.f);
    //! @}

    //! @brief Input keypoints.
    //! @{
    auto keys1() const -> const KeypointList<OERegion, float>&
    {
      return _keys1;
    }

    auto keys2() const -> const KeypointList<OERegion, float>&
    {
      return _keys2;
    }
    //! @}

    //! @{
    //! @brief Return matches.
    DO_SARA_EXPORT
    std::vector<Match> compute_matches();

    std::vector<Match> compute_self_matches()
    {
      return compute_matches();
    }
    //! @}

  private: /* data members */
    //! Input parameters.
    const KeypointList<OERegion, float>& _keys1;
    const KeypointList<OERegion, float>& _keys2;
    float _squared_ratio_thres;
    //! Internals.
    KeyProximity _is_too_close;
    std::size_t _max_neighbors;
    std::vector<int> _vec_indices;
    std::vector<float> _vec_dists;
    bool _self_matching;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
