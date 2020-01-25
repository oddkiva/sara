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

  class StudyAffineMetric : public StudyOnMikolajczykDataset
  {
  public:
    StudyAffineMetric(const std::string& absParentFolderPath,
                      const std::string& name, const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
    {
    }
    bool operator()(float inlierThres, float squaredEll, size_t K,
                    double squaredRhoMin);
  };

} /* namespace DO::Sara */
