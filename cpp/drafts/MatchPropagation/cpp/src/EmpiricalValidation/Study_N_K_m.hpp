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

  class Study_N_K_m : public StudyOnMikolajczykDataset
  {
  public:
    Study_N_K_m(const std::string& absParentFolderPath, const std::string& name,
                const std::string& featType)
      : StudyOnMikolajczykDataset(absParentFolderPath, name, featType)
    {
    }
    bool operator()(float inlier_thres, float squared_ell, size_t K,
                    double squaredRhoMin);

  private:
    void getStat(Statistics& stat_N_K, Statistics& stat_hatN_K, Statistics& stat_diff,
                 const std::vector<std::vector<size_t>>& N_K,
                 const std::vector<std::vector<size_t>>& hatN_K);

    void getStat(Statistics& stat_N_K, Statistics& stat_hatN_K, Statistics& stat_diff,
                 const std::vector<size_t>& indices,
                 const std::vector<std::vector<size_t>>& N_K,
                 const std::vector<std::vector<size_t>>& hatN_K);


    void checkNeighborhood(const std::vector<std::vector<size_t>>& N_K,
                           const std::vector<Match>& M,
                           const PairWiseDrawer& drawer);

    bool saveStats(const std::string& name, const std::vector<Statistics>& stat_N_Ks,
                   const std::vector<Statistics>& stat_hatN_Ks,
                   const std::vector<Statistics>& stat_diffs);
  };

}  // namespace DO::Sara
