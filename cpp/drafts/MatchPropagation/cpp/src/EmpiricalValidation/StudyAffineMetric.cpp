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

#include "StudyAffineMetric.hpp"

#include <DO/Sara/Match/PairWiseDrawer.hpp>

using namespace std;


namespace DO::Sara {

  bool StudyAffineMetric::operator()(float inlier_thres, float squared_ell)
  {
    for (int j = 1; j < 6; ++j)
    {
      // View the image pair.
      open_window_for_image_pair(0, j);
      PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
      drawer.set_viz_params(1.0f, 1.0f, PairWiseDrawer::CatH);
      drawer.display_images();
      {
        // Read the set of keypoints $\mathcal{X}$ in image 1.
        const auto& X = dataset().keys(0);
        // Read the set of keypoints $\mathcal{Y}$ in image 2.
        const auto& Y = dataset().keys(j);
        // Compute initial matches.
        const auto M = compute_matches(X, Y, squared_ell);
        //
        vector<size_t> inliers, outliers;
        get_inliers_and_outliers(inliers, outliers, M, dataset().H(j),
                                 inlier_thres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
      }
      close_window_for_image_pair();
    }

    const auto folder = string_src_path(dataset().name());
    mkdir(folder);

    return true;
  }

}  // namespace DO::Sara
