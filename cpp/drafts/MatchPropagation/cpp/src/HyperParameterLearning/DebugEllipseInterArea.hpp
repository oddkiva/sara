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

  class DebugEllipseInterArea : public StudyOnMikolajczykDataset
  {
  public:
    DebugEllipseInterArea(const std::string& absParentFolderPath,
                          const std::string& name, const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
      , debug_(true)
    {
    }
    bool operator()(float inlierThres, float squaredEll);

  private:
    void checkReprojectedEllipse(const Match& m, const PairWiseDrawer& drawer,
                                 Ellipse& y, Ellipse& H_Sx,
                                 double polyApproxOverlap,
                                 double analyticalOverlap, double angle_phi_ox,
                                 double angle_y, double error) const;

    bool saveStats(const std::string& name,
                   const std::vector<Stat>& stat_overlaps,
                   const std::vector<Stat>& stat_angles) const;

  private:
    bool debug_;
  };

} /* namespace DO::Sara */
