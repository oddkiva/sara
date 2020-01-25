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

#include "../HyperParameterLearning/Stat.hpp"

#include <DO/Sara/FileSystem.hpp>

namespace DO {

  class StudyOnMikolajczykDataset
  {
  public:
    // Index Distance Pair
    typedef std::pair<size_t, float> IndexDist;
    struct CompareIndexDist
    {
      bool operator()(const IndexDist& p1, const IndexDist& p2) const
      { return p1.second < p2.second; }
    };

    // Constructor
    StudyOnMikolajczykDataset(const std::string& absParentFolderPath,
                              const std::string& name,
                              const std::string& featType);

    // Viewing, convenience functions...
    const MikolajczykDataset& dataset() const { return dataset_; }
    void openWindowForImagePair(size_t i, size_t j) const;
    void closeWindowForImagePair() const;
    // Match related functions.
    std::vector<Match> computeMatches(const Set<OERegion, RealDescriptor>& X,
                                      const Set<OERegion, RealDescriptor>& Y,
                                      float squaredEll) const;
    void getInliersAndOutliers(std::vector<size_t>& inliers,
                               std::vector<size_t>& outliers,
                               const std::vector<Match>& matches,
                               const Matrix3f& H,
                               float thres) const;
    std::vector<IndexDist> sortMatchesByReprojError(const std::vector<Match>& M,
                                                    const Matrix3f& H) const;
    std::vector<size_t> getMatches(const std::vector<IndexDist>& sortedMatches, 
                                   float reprojLowerBound,
                                   float reprojUpperBound) const;
    std::vector<size_t> getMatches(const std::vector<Match>& M,
                                   const Matrix3f& H,
                                   float reprojLowerBound,
                                   float reprojUpperBound) const
    {
      return getMatches(sortMatchesByReprojError(M, H),
                        reprojLowerBound, reprojUpperBound);
    }

  private:
    MikolajczykDataset dataset_;
  };

} /* namespace DO */
