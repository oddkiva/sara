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

#include "StudyPerfWithHatN_K.hpp"
#include "../GrowMultipleRegions.hpp"

using namespace std;

namespace DO::Sara {

  bool
  StudyPerfWithHat_N_K::
  operator()(float squared_ell, size_t numRegionGrowths,
             size_t K, double rho_min)
  {
    // ====================================================================== //
    /* Below: Mikolajczyk et al.'s parameter in their IJCV 2005 paper. 
     *
     * Let (x,y) be a match. It is an inlier if it satisfies:
     * $$\| \mathbf{H} \mathbf{x} - \mathbf{y} \|_2 < 1.5 \ \textrm{pixels}$$
     *
     * where $\mathbf{H}$ is the ground truth homography.
     * 1.5 pixels is used in the above-mentioned paper.
     */
    float mikolajczykInlierThres = 1.5f;    
    // Set of thresholds.
    vector<float> thres;
    thres.push_back(mikolajczykInlierThres);
    thres.push_back(5.f);

    for (int j = 1; j < 6; ++j)
    {
      PairWiseDrawer *drawer = 0;
      if (_display)
      {
        // View the image pair.
        drawer = new PairWiseDrawer(dataset().image(0), dataset().image(j));
        open_window_for_image_pair(0, j);
        drawer->set_viz_params(1.0f, 1.0f, PairWiseDrawer::CatH);
        drawer->display_images();
      }

      // The job is here.
      {
        // Read the set of keypoints $\mathcal{X}$ in image 1.
        const KeypointList<OERegion, float>& X = dataset().keys(0);
        // Read the set of keypoints $\mathcal{Y}$ in image 2.
        const KeypointList<OERegion, float>& Y = dataset().keys(j);
        // Compute initial matches $\mathcal{M}$.
        vector<Match> M(compute_matches(X, Y, squared_ell));
        // Get ground truth homography
        const Matrix3f& H = dataset().H(j);

        for (size_t t = 0; t != thres.size(); ++t)
        {
          bool success;
          success = run(M, H, j, squared_ell,
                             thres[t],
                             numRegionGrowths, K, rho_min, false, drawer);
          success = run(M, H, j, squared_ell,
                             thres[t],
                             numRegionGrowths, K, rho_min, true, drawer);
          if (!success)
          {
            if (_display)
            {
              close_window_for_image_pair();
              if (drawer)
                delete drawer;
            }
            return false;
          }
        }
      }
      if (_display)
      {
        close_window_for_image_pair();
        if (drawer)
          delete drawer;
      }
    }

    
    return true;
  }

  bool
  StudyPerfWithHat_N_K::
  run(const vector<Match>& M,
           const Matrix3f& H, size_t imgIndex,
           float squared_ell, float inlier_thres,
           size_t numRegionGrowths,
           size_t K, double rho_min,
           bool useHatN_K,
           const PairWiseDrawer *drawer) const
  {
    string comment;
    comment  = "Evaluating outlier resistance on dataset '";
    comment += dataset().name() + ":\n\tpair 1-"+to_string(imgIndex+1);
    comment += (useHatN_K ? "\n\thatN_K" : "\n\tN_K");
    comment += "\n\tfeatType = " + dataset().featType();
    comment += "\n\tsquaredEll = " + to_string(squared_ell);
    comment += "\n\tK = " + to_string(K);
    comment += "\n\trho_min = " + to_string(rho_min);
    comment += "\n\tinlierThres = " + to_string(inlier_thres);
    print_stage(comment);

    // Get subset of matches.
    vector<size_t> inliers, outliers;
    get_inliers_and_outliers(inliers, outliers, M, H, inlier_thres);
    // We want to perform our analysis on this particular subset of matches of interest.
    bool verbose = _debug && (drawer != nullptr);
    RegionGrowingAnalyzer analyzer(M, H, verbose);
    analyzer.set_inliers(inliers);

    // Grow multiple regions.
    cout << "Growing Regions... ";
    GrowthParams params(K, rho_min);
    GrowMultipleRegions growMultipleRegions(M, params, _debug ? 1 : 0);
    if (useHatN_K)
      growMultipleRegions.build_hat_N_Ks();
    vector<Region> RR(growMultipleRegions(numRegionGrowths, &analyzer, drawer));
    cout << "Done!" << endl;

    // Compute the statistics.
    cout << "Computing stats... ";
    // Get found matches in a proper container.
    vector<size_t> all_matches;
    {
      Region allR;
      for (size_t i = 0; i != RR.size(); ++i)
        for (Region::iterator j = RR[i].begin(); j != RR[i].end(); ++j)
          allR.insert(*j);
      all_matches.reserve(allR.size());
      for (Region::iterator i = allR.begin(); i != allR.end(); ++i)
        all_matches.push_back(*i);
    }
    analyzer.compute_positives_and_negatives(all_matches);


    // Save stats.
    cout << "Saving stats... ";
    string folder; 
    folder = dataset().name()+"/performance_hat_N_K";
    folder = string_src_path(folder);
#pragma omp critical
    {
      mkdir(folder);
    }

    string neighborhoodName = useHatN_K ? "_HatN_K_" : "_N_K_";

    const string namePrecRecall("prec_recall_"
                              + dataset().name() 
                              + "_" + to_string(1) + "_" + to_string(imgIndex+1)
                              + neighborhoodName
                              + "_sqEll_" + to_string(squared_ell)
                              + "_nReg_ " + to_string(numRegionGrowths)
                              + "_K_"  + to_string(K)
                              + "_rhoMin_" + to_string(rho_min)
                              + "_inlierThres_" + to_string(inlier_thres)
                              + dataset().featType()
                              + ".txt");

    const string nameStatRegions("stat_regions_"
                               + dataset().name() 
                               + "_" + to_string(1) + "_" + to_string(imgIndex+1)
                               + neighborhoodName
                               + "_sqEll_" + to_string(squared_ell)
                               + "_nReg_ " + to_string(numRegionGrowths)
                               + "_K_"  + to_string(K)
                               + "_rhoMin_" + to_string(rho_min)
                               + "_inlierThres_" + to_string(inlier_thres)
                               + dataset().featType()
                               + ".txt");

    bool success;
#pragma omp critical 
    {
      success = analyzer.save_precision_recall_etc(string(folder+"/"+namePrecRecall)) &&
                analyzer.save_region_number_statistics(string(folder+"/"+nameStatRegions));
    }
    if (!success)
    {
      cerr << "Could not save stats:" << endl
           << string(folder+"/"+namePrecRecall)  << endl
           << string(folder+"/"+nameStatRegions) << endl;
      return false;
    }
    cout << "Done!" << endl;
    return true; 
  }

} /* namespace DO */
