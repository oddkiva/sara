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

  class StudyPerfWithHat_N_K : public StudyOnMikolajczykDataset
  {
  public:
    StudyPerfWithHat_N_K(const std::string& absParentFolderPath,
                         const std::string& name, const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
      , debug_(false)
      , display_(false)
    {
    }
    bool operator()(float squaredEll, size_t numRegionGrowths, size_t K,
                    double rho_min);

  private:
    bool doTheJob(const std::vector<Match>& M, const Matrix3f& H,
                  size_t imgIndex, float squaredEll, float inlierThres,
                  size_t numRegionGrowths, size_t K, double rho_min,
                  bool useHatN_K, const PairWiseDrawer* pDrawer = 0) const;

  private:
    bool debug_;
    bool display_;
  };

}  // namespace DO::Sara
