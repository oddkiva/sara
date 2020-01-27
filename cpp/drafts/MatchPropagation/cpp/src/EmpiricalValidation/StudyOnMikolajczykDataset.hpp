// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
/*!
 *  This file implements a part of the method published in:
 *
 *  Efficient and Scalable 4th-order Match Propagation
 *  David Ok, Renaud Marlet, and Jean-Yves Audibert.
 *  ACCV 2012, Daejeon, South Korea.
 */

#pragma once

#include "MikolajczykDataset.hpp"

#include "../Statistics.hpp"

#include <DO/Sara/FileSystem.hpp>


namespace DO::Sara {

  class DO_SARA_EXPORT StudyOnMikolajczykDataset
  {
  public:
    // Index Distance Pair
    typedef std::pair<size_t, float> IndexDist;
    struct CompareIndexDist
    {
      bool operator()(const IndexDist& p1, const IndexDist& p2) const
      {
        return p1.second < p2.second;
      }
    };

    // Constructor
    StudyOnMikolajczykDataset(const std::string& abs_parent_folder_path,
                              const std::string& name,
                              const std::string& feature_type);

    // Viewing, convenience functions...
    const MikolajczykDataset& dataset() const
    {
      return _dataset;
    }

    void open_window_for_image_pair(size_t i, size_t j) const;

    void close_window_for_image_pair() const;

    // Match related functions.
    std::vector<Match> compute_matches(const KeypointList<OERegion, float>& X,
                                       const KeypointList<OERegion, float>& Y,
                                       float squared_ell) const;

    void get_inliers_and_outliers(std::vector<size_t>& inliers,
                                  std::vector<size_t>& outliers,
                                  const std::vector<Match>& matches,
                                  const Matrix3f& H, float thres) const;

    std::vector<IndexDist>
    sort_matches_by_reprojection_error(const std::vector<Match>& M,
                                       const Matrix3f& H) const;

    std::vector<size_t> get_matches(const std::vector<IndexDist>& sorted_matches,
                                    float reproj_lower_bound,
                                    float reproj_upper_bound) const;

    std::vector<size_t> get_matches(const std::vector<Match>& M,
                                    const Matrix3f& H, float reproj_lower_bound,
                                    float reproj_upper_bound) const
    {
      return get_matches(sort_matches_by_reprojection_error(M, H),
                         reproj_lower_bound, reproj_upper_bound);
    }

  private:
    MikolajczykDataset _dataset;
  };

}  // namespace DO::Sara
