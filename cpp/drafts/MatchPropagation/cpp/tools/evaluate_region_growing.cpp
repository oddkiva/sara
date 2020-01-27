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

#include <DO/Sara/SfM/Detectors/SIFT.hpp>
#include "StudyOnMikolajczykDataset.hpp"
#include "MatchPropagation.hpp"


using namespace std;
using namespace DO::Sara;


class TestGrowRegion : public StudyOnMikolajczykDataset
{
public:
  TestGrowRegion(const string& abs_parent_folder_path,
                 const string& name,
                 const string& feature_type)
    : StudyOnMikolajczykDataset(abs_parent_folder_path, name, feature_type)
  {}

  void operator()()
  {
    float ell = 1.0f;
    float inlierThres = 5.f;
    size_t K = 200;
    double rho_min = 0.5;
    //
    double angleDeg1 = 15;
    double angleDeg2 = 25;
    //
    bool displayInliers = false;

    for (int j = 5; j < 6; ++j)
    {
      // View the image pair.
      open_window_for_image_pair(0, j);
      PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
      drawer.set_viz_params(1.0f, 1.0f, PairWiseDrawer::CatH);
      drawer.display_images();
      {
        // Set of keypoints $\mathcal{X}$ in image 1.
        const auto& X = dataset().keys(0);
        // Set of keypoints $\mathcal{Y}$ in image 2.
        const auto& Y = dataset().keys(j);
        // Ground truth homography from image 1 to image 2.
        const Matrix3f& H = dataset().H(j);
        // Compute initial matches.
        vector<Match> M(compute_matches(X, Y, ell*ell));
        // Get inliers and outliers.
        vector<size_t> inliers, outliers;
        get_inliers_and_outliers(inliers, outliers, M, H, inlierThres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        // View inliers.
        if (displayInliers)
        {
          for (size_t i = 0; i != inliers.size(); ++i)
            drawer.draw_match(M[inliers[i]], Cyan8);
          get_key();
          drawer.display_images();
        }

        RegionGrowingAnalyzer analyzer(M, H);
        analyzer.set_subset_of_interest(inliers);

        // Grow region from the first seed
        size_t seed = inliers[0];
        GrowthParams growthParams(K, rho_min, angleDeg1, angleDeg2);
        DynamicMatchGraph G(M, growthParams.K(), growthParams.rho_min());
        GrowRegion growRegion(seed, G, growthParams);
        Region R(growRegion(numeric_limits<size_t>::max(), &drawer, &analyzer));

        analyzer.compute_local_affine_consistency_statistics();
        /*string aff_stats_name = "local_aff_stat_" + to_string(1) + "_" + to_string(j+1)
                              + dataset().feature_type()
                              + ".txt";
        aff_stats_name = string_src_path(aff_stats_name);
        analyzer.saveLocalAffineConsistencyStats(aff_stats_name);*/

        string dR_stat_name = "evol_dR_size_"
                            + to_string(1) + "_" + to_string(j+1)
                            + "_ell_" + to_string(ell)
                            + dataset().feature_type()
                            + ".txt";
        dR_stat_name = string_src_path(dR_stat_name);
        analyzer.save_boundary_region_evolution(dR_stat_name);
      }
      close_window_for_image_pair();
    }
  }
};

class TestGrowMultipleRegions : public StudyOnMikolajczykDataset
{
public:
  TestGrowMultipleRegions(const string& abs_parent_folder_path,
                          const string& name,
                          const string& feature_type)
    : StudyOnMikolajczykDataset(abs_parent_folder_path, name, feature_type)
  {}

  void operator()()
  {
    float ell = 1.0f;
    float inlierThres = 5.f;
    size_t K = 200;
    double rho_min = 0.3;
    //
    double angleDeg1 = 15;
    double angleDeg2 = 25;
    //
    bool displayInliers = false;

    for (int j = 4; j < 6; ++j)
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
        vector<Match> M(compute_matches(X, Y, ell*ell));
//#define REDUNDANCY
#ifdef REDUNDANCY
        print_stage("Removing redundant matches");
        // Get the redundancy components.
        vector<vector<size_t> > components;
        vector<size_t> representers;
        double thres = 3.0;
        ComputeN_K eliminateRedundancies(M, 1e3);
        eliminateRedundancies(components, representers, M, thres);
        // Only keep the best representers.
        vector<Match> filteredM(representers.size());
        for (size_t i = 0; i != filteredM.size(); ++i)
          filteredM[i] = M[representers[i]];
#else
        vector<Match>& filteredM = M;
#endif
        // Get inliers and outliers.
        vector<size_t> inliers, outliers;
        get_inliers_and_outliers(inliers, outliers, filteredM, dataset().H(j), inlierThres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        // View inliers.
        if (displayInliers)
        {
          for (size_t i = 0; i != inliers.size(); ++i)
            drawer.draw_match(M[inliers[i]], Cyan8);
          get_key();
          drawer.display_images();
        }

        // Grow multiple regions.
        size_t N = 5000;
        GrowthParams params(K, rho_min, angleDeg1, angleDeg2);
        int verbose = 2;
        GrowMultipleRegions growMultipleRegions(M, params, verbose);
        //growMultipleRegions.buildHatN_Ks();
        vector<Region> RR(growMultipleRegions(N, 0, &drawer));
      }
      close_window_for_image_pair();
    }
  }
};


void testOnImage(const string& file1, const string& file2)
{
  Image<Rgb8> image1, image2;
  load(image1, file1);
  load(image2, file2);

  // View the image pair.
  print_stage("Display image pair and the features");
  float scale = 1.f;
  int w = int((image1.width()+image2.width())*scale);
  int h = int(max(image1.height(), image2.height())*scale);
  openWindow(w, h);
  setAntialiasing(activeWindow());

  // Setup viewing.
  PairWiseDrawer drawer(image1, image2);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);
  drawer.display_images();
  get_key();

  // Compute keypoints.
  const auto keys1 = DoGSiftDetector().run(image1.convert<unsigned char>());
  const auto keys2 = DoGSiftDetector().run(image2.convert<unsigned char>());
  cout << "Image 1: " << keys1.size() << " keypoints" << endl;
  cout << "Image 2: " << keys2.size() << " keypoints" << endl;

  // Compute initial matches
  float ell = 1.0f;
  AnnMatcher matcher(keys1, keys2, ell);
  vector<Match> M = matcher.compute_matches();
  cout << M.size() << " matches" << endl;

  // Growing parameters.
  size_t K = 80;
  double rho_min = 0.3;
  //
  double angleDeg1 = 15;
  double angleDeg2 = 25;
  //
  size_t N = 1000;
  GrowthParams params(K, rho_min, angleDeg1, angleDeg2);
  // Grow multiple regions.
  int verbose = 2;
  GrowMultipleRegions growMultipleRegions(M, params, verbose);
  vector<Region> RR(growMultipleRegions(N, 0, &drawer));
  saveScreen(activeWindow(), srcPath("result.png"));
}


int main()
{
  // Dataset paths.
  const auto mikolajczyk_dataset_folder = string("Mikolajczyk/");
  cout << mikolajczyk_dataset_folder << endl;
  const string folders[8] = { 
    "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall" 
  };
  const string ext[4] = { ".dog", ".haraff", ".hesaff", ".mser" };

  // Select the test module you want to run.
  bool test_growRegionFromBestSeed = false;
  bool test_growMultipleRegions = false;
  size_t dataset = 0;
  size_t ext_index = 0;

  // Call the desired modules.
  if (test_growRegionFromBestSeed)
  {
    TestGrowRegion testGrowRegion(mikolajczyk_dataset_folder,
                                  folders[dataset], ext[ext_index]);
    testGrowRegion();
  }
  if (test_growMultipleRegions)
  {
    TestGrowMultipleRegions 
      testGrowMultipleRegions(mikolajczyk_dataset_folder,
                              folders[dataset], ext[ext_index]);
    testGrowMultipleRegions();
  }

  testOnImage(mikolajczyk_dataset_folder + "bark/img1.ppm",
              mikolajczyk_dataset_folder + "bark/img4.ppm");
  
  return 0;
}
