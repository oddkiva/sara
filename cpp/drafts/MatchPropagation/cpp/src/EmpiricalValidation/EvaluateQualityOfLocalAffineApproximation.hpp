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

  class EvalQualityOfLocalAffApprox : public StudyOnMikolajczykDataset
  {
  public:
    EvalQualityOfLocalAffApprox(const std::string& absParentFolderPath,
                                const std::string& name,
                                const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
      , _debug(false)
      , _display(false)
    {
    }
    bool operator()(float squaredEll, size_t numRegionGrowths, size_t K,
                    double rho_min) const;

  private:
    bool run(const std::vector<Match>& M, const std::vector<IndexDist>& sortedM,
             const Matrix3f& H, size_t imgIndex, float squaredEll, float lb,
             float ub, size_t numRegionGrowths, size_t K, double rho_min,
             const PairWiseDrawer* pDrawer = 0) const;

  private:
    bool _debug;
    bool _display;
  };

}  // namespace DO::Sara
