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

  //! @brief This class evaluates the quality of the local affine approximation.
  /*! 
   *  Mikolajczyk et al.'s parameter in their IJCV 2005 paper.
   *
   *  Let (x,y) be a match. It is an inlier if it satisfies:
   *  $$\| \mathbf{H} \mathbf{x} - \mathbf{y} \|_2 < 1.5 \ \textrm{pixels}$$
   *
   *  where $\mathbf{H}$ is the ground truth homography.
   *  1.5 pixels is used in the paper.
   *
   *  This class enables us to do the same thing but for various thresholds.
   *
   */
  class DO_SARA_EXPORT EvaluateQualityOfLocalAffineApproximation
    : public StudyOnMikolajczykDataset
  {
  public:
    EvaluateQualityOfLocalAffineApproximation(
        const std::string& abs_parent_folder_path, const std::string& name,
        const std::string& feature_type)
      : StudyOnMikolajczykDataset(abs_parent_folder_path, name, feature_type)
    {
    }
    bool operator()(float squared_ell, size_t numRegionGrowths, size_t K,
                    double rho_min) const;

  private:
    bool run(const std::vector<Match>& M,
             const std::vector<IndexDist>& M_sorted, const Matrix3f& H,
             size_t image_index, float squared_ell, float lb, float ub,
             size_t numRegionGrowths, size_t K, double rho_min,
             const PairWiseDrawer* drawer = 0) const;

  private:
    bool _debug{false};
    bool _display{false};
  };

}  // namespace DO::Sara
