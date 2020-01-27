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

  class DO_SARA_EXPORT StudyRegionFusion : public StudyOnMikolajczykDataset
  {
  public:
    StudyRegionFusion(const std::string& abs_parent_folder,
                      const std::string& name, const std::string& feature_type)
      : StudyOnMikolajczykDataset(abs_parent_folder, name, feature_type)
    {
    }

    bool operator()(float inlier_thres, float squared_ell, size_t K,
                    double squared_rho_min);
  };

}  // namespace DO::Sara
