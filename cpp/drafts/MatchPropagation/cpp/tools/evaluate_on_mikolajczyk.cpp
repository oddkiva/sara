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

#include "StudyOnMikolajczykDataset.hpp"
#include <DO/Sara/Match/PairWiseDrawer.hpp>

using namespace std;
using namespace DO::Sara;


// Dataset paths.
const string mikolajczyk_directory = "Mikolajczyk";
const string folders[8] = {"bark",   "bikes", "boat", "graf",
                           "leuven", "trees", "ubc",  "wall"};
const string ext[4] = {".dog", ".haraff", ".hesaff", ".mser"};


class TestRegion : public StudyOnMikolajczykDataset
{
public:
  TestRegion(const string& abs_parent_folder_path, const string& name,
             const string& feature_type)
    : StudyOnMikolajczykDataset(abs_parent_folder_path, name, feature_type)
  {
  }

  void operator()()
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
        vector<Match> M(compute_matches(X, Y, 1.2f * 1.2f));
        // Get inliers and outliers.
        vector<size_t> inliers, outliers;
        get_inliers_and_outliers(inliers, outliers, M, dataset().H(j), 1.5f);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        // View inliers.
        for (size_t i = 0; i != inliers.size(); ++i)
          drawer.draw_match(M[inliers[i]]);
        get_key();
      }
      close_window_for_image_pair();
    }
  }
};

GRAPHICS_MAIN()
{
  cout << mikolajczyk_directory << endl;
  TestRegion test(mikolajczyk_directory, folders[0], ext[1]);
  test();
  return 0;
}
