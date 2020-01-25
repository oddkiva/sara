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

#include "StudyOnMikolajczykDataset.hpp"

namespace DO::Sara {

  class LearnPf : public StudyOnMikolajczykDataset
  {
  public:
    LearnPf(const std::string& absParentFolderPath,
            const std::string& name,
            const std::string& featType,
            bool approxEllInterArea = false)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
      , debug_(false)
      , display_(false)
      , approx_ell_inter_area_(approxEllInterArea)
    {}
    bool operator()(float squaredEll) const;

  private:
    bool saveStats(const std::string& name,
                   const std::vector<Stat>& stat_overlaps,
                   const std::vector<Stat>& stat_angles) const;

    void doTheJob(Stat& stat_overlap, Stat& stat_angle,
                  const std::vector<Match>& M,
                  const std::vector<IndexDist>& sortedM,
                  const Matrix3f& H,
                  float lb, float ub,
                  const PairWiseDrawer *pDrawer = 0) const;

  private:
    bool debug_;
    bool display_;
    //bool find_dilation_;
    bool approx_ell_inter_area_;
  };

} /* namespace DO::Sara */
