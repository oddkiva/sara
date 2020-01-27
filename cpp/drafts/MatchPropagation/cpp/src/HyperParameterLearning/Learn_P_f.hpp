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

#include "../EmpiricalValidation/StudyOnMikolajczykDataset.hpp"
#include "../Statistics.hpp"

#include <DO/Sara/Match/PairWiseDrawer.hpp>


namespace DO::Sara {

  class DO_SARA_EXPORT LearnPf : public StudyOnMikolajczykDataset
  {
  public:
    LearnPf(const std::string& abs_parent_folder_path, const std::string& name,
            const std::string& feature_type, bool approx_ell_inter_area = false)
      : StudyOnMikolajczykDataset(abs_parent_folder_path, name, feature_type)
      , debug_(false)
      , display_(false)
      , approx_ell_inter_area_(approx_ell_inter_area)
    {
    }
    bool operator()(float squared_ell) const;

  private:
    bool save_statistics(const std::string& name,
                         const std::vector<Statistics>& stat_overlaps,
                         const std::vector<Statistics>& stat_angles) const;

    void run(Statistics& stat_overlap, Statistics& stat_angle,
             const std::vector<Match>& M, const std::vector<IndexDist>& M_sorted,
             const Matrix3f& H, float lb, float ub,
             PairWiseDrawer* drawer = 0) const;

  private:
    bool debug_;
    bool display_;
    bool approx_ell_inter_area_;
  };

} /* namespace DO::Sara */
