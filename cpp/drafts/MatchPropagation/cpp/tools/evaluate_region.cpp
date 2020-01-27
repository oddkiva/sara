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
#include "Region.hpp"
#include "RegionBoundary.hpp"


using namespace std;
using namespace DO::Sara;


class TestRegion : public StudyOnMikolajczykDataset
{
public:
  TestRegion(const string& abs_parent_folder_path,
             const string& name,
             const string& feature_type)
    : StudyOnMikolajczykDataset(abs_parent_folder_path, name, feature_type)
  {}

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

        // Region
        Region R;
        for (size_t i = 0; i != inliers.size(); ++i)
        {
          //cout << "Inserting M["<<inliers[i]<<"]" << endl;
          R.insert(M[inliers[i]], M);
          //cout << "Is M["<<inliers[i]<<"] correctly inserted?" << endl;
          if (!R.find(M[inliers[i]], M))
          {
            cerr << "Cannot find match:\n" << M[inliers[i]] << endl;
            cerr << "Index is = " << inliers[i] << endl;
            break;
          }
          /*if (R.find(M[inliers[i]], M))
            cout << "M["<<inliers[i]<<"] correctly inserted" << endl;*/
        }
        cout << "R.size() = " << R.size() << endl;
        get_key();

      }
      close_window_for_image_pair();
    }
  }
};

class TestRegionBoundary : public StudyOnMikolajczykDataset
{
public:
  TestRegionBoundary(const string& abs_parent_folder_path,
                     const string& name,
                     const string& feature_type)
    : StudyOnMikolajczykDataset(abs_parent_folder_path, name, feature_type)
  {}

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
        vector<Match> M(compute_matches(X, Y, 1.2f*1.2f));

        // Get inliers and outliers.
        vector<size_t> inliers, outliers;
        get_inliers_and_outliers(inliers, outliers, M, dataset().H(j), 1.5f);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;

        // View inliers.
        for (size_t i = 0; i != inliers.size(); ++i)
          drawer.draw_match(M[inliers[i]]);
        
        // ================================================================== //
        // Region boundary.
        // Testing insertion and query.
        RegionBoundary dR(M);
        for (size_t i = 0; i != inliers.size(); ++i)
        {
          //cout << "Inserting M["<<inliers[i]<<"]" << endl;
          dR.insert(M[inliers[i]]);
          //cout << "Is M["<<inliers[i]<<"] correctly inserted?" << endl;
          if (!dR.find(M[inliers[i]]))
          {
            cerr << "Cannot find match:\n" << M[inliers[i]] << endl;
            cerr << "Index is = " << inliers[i] << endl;
            break;
          }
          if (dR.find(M[inliers[i]]))
            cout << "M["<<inliers[i]<<"] correctly inserted" << endl;
          cout << "dR.size() = " << dR.size() << endl;
        }
        cout << "dR.size() = " << dR.size() << endl;
        get_key();

        // Testing iterators.
        for (RegionBoundary::const_iterator m = dR.begin(); m != dR.end(); ++m)
        {
          cout << "M[" << m.index() << "] = \n" << *m << endl;
          drawer.draw_match(*m, Red8);
        }
        get_key();

        // Testing erasing.
        RegionBoundary dR2(M);
        dR2.insert(inliers[0]);
        // Testing iterators.
        for (RegionBoundary::const_iterator m = dR2.begin(); m != dR2.end(); ++m)
        {
          cout << "M[" << m.index() << "] = \n" << *m << endl;
          drawer.draw_match(*m, Red8);
        }
        get_key();
        dR2.erase(inliers[0]);
        cout << "dR2.size() = " << dR2.size() << endl;
        get_key();
      }
      close_window_for_image_pair();
    }
  }
};


GRAPHICS_MAIN()
{
  // Dataset paths.
  const string mikolajczyk_dataset_folder = "mikolajczyk";
  const string folders[8] = { 
    "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall" 
  };
  const string ext[4] = { ".dog", ".haraff", ".hesaff", ".mser" };
  TestRegion testRegion(mikolajczyk_dataset_folder, folders[0], ext[0]);
  testRegion();

  TestRegionBoundary testRegionBoundary(mikolajczyk_dataset_folder, folders[0], ext[0]);
  testRegionBoundary();
  return 0;
}
