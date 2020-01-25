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

using namespace std;


namespace DO::Sara {

  bool StudyAffineMetric::operator()(float inlierThres, float squaredEll,
                                     size_t K, double squaredRhoMin)
  {
    for (int j = 1; j < 6; ++j)
    {
      // View the image pair.
      openWindowForImagePair(0, j);
      PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
      drawer.setVizParams(1.0f, 1.0f, PairWiseDrawer::CatH);
      drawer.displayImages();
      {
        // Read the set of keypoints $\mathcal{X}$ in image 1.
        const vector<Keypoint>& X = dataset().keys(0);
        // Read the set of keypoints $\mathcal{Y}$ in image 2.
        const vector<Keypoint>& Y = dataset().keys(j);
        // Compute initial matches.
        vector<Match> M(computeMatches(X, Y, squaredEll));
        //
        vector<size_t> inliers, outliers;
        getInliersAndOutliers(inliers, outliers, M, dataset().H(j),
                              inlierThres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
      }
      closeWindowForImagePair();
    }

    string folder(dataset().name());
    folder = stringSrcPath(folder);
    createDirectory(folder);

    return true;
  }

}  // namespace DO::Sara
