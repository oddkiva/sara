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

#include <DO/Sara/Graphics/Match/PairWiseDrawer.hpp>


namespace DO::Sara {

  class DO_SARA_EXPORT StudyParametersOfMatchNeighborhood
    : public StudyOnMikolajczykDataset
  {
  public:
    StudyParametersOfMatchNeighborhood(
        const std::string& abs_parent_folder_path, const std::string& name,
        const std::string& feature_type)
      : StudyOnMikolajczykDataset(abs_parent_folder_path, name, feature_type)
    {
    }
    bool operator()(float inlier_thres, float squared_ell, size_t K,
                    double squared_rho_min);

  private:
    void check_neighborhood(const std::vector<std::vector<size_t>>& N_K,
                            const std::vector<Match>& M,
                            const PairWiseDrawer& drawer);

    void compute_statistics(Statistics& stat_N_K, Statistics& stat_hatN_K,
                            Statistics& stat_diff,
                            const std::vector<std::vector<size_t>>& N_K,
                            const std::vector<std::vector<size_t>>& hatN_K);

    void compute_statistics(Statistics& stat_N_K, Statistics& stat_hatN_K,
                            Statistics& stat_diff,
                            const std::vector<size_t>& indices,
                            const std::vector<std::vector<size_t>>& N_K,
                            const std::vector<std::vector<size_t>>& hatN_K);


    bool compute_statistics(const std::string& name,
                            const std::vector<Statistics>& stat_N_Ks,
                            const std::vector<Statistics>& stat_hatN_Ks,
                            const std::vector<Statistics>& stat_diffs);
  };

}  // namespace DO::Sara
