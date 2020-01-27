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

#include <DO/Sara/Geometry/Objects/Ellipse.hpp>
#include <DO/Sara/Match/PairWiseDrawer.hpp>


namespace DO::Sara {

  class DebugEllipseInterArea : public StudyOnMikolajczykDataset
  {
  public:
    DebugEllipseInterArea(const std::string& abs_parent_folder_path,
                          const std::string& name, const std::string& feature_type)
      : StudyOnMikolajczykDataset(abs_parent_folder_path, name, feature_type)
      , _debug(true)
    {
    }
    bool operator()(float inlier_thres, float squared_ell);

  private:
    void check_reprojected_ellipse(const Match& m, const PairWiseDrawer& drawer,
                                   Ellipse& y, Ellipse& H_Sx,
                                   double polygonal_overlap,
                                   double analytical_overlap,
                                   double angle_phi_ox, double angle_y,
                                   double error) const;

    bool save_statistics(const std::string& name,
                         const std::vector<Statistics>& stat_overlaps,
                         const std::vector<Statistics>& stat_angles) const;

  private:
    bool _debug;
  };

} /* namespace DO::Sara */
