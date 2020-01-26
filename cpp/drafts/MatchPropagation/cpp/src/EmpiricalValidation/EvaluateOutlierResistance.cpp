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

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "EvaluateOutlierResistance.hpp"
#include "GrowMultipleRegions.hpp"
#include "MatchNeighborhood.hpp"


using namespace std;


namespace DO::Sara {

  bool EvaluateOutlierResistance::operator()(float squared_ell,
                                         size_t num_region_growths, size_t K,
                                         size_t k, double rho_min) const
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
    const auto mikolajczyk_inlier_thres = 1.5f;
    // Set of thresholds.
    auto thres = vector<float>{};
    thres.push_back(mikolajczyk_inlier_thres);
    thres.push_back(5.f);

    // ====================================================================== //
    // Let's go.
    for (int j = 1; j < 6; ++j)
    {
      auto drawer = unique_ptr<PairWiseDrawer>{};
      if (_display)
      {
        // View the image pair.
        drawer.reset(
            new PairWiseDrawer(dataset().image(0), dataset().image(j)));
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
          success = run(M, H, j, squared_ell, thres[t], num_region_growths, K, k,
                        rho_min, drawer.get());
          if (!success)
          {
            if (_display)
              close_window_for_image_pair();

            return false;
          }
        }
      }

      if (_display)
        close_window_for_image_pair();
    }

    return true;
  }

  bool EvaluateOutlierResistance::run(const vector<Match>& M,
                                       const Matrix3f& H, size_t img_index,
                                       float squared_ell, float inlier_thres,
                                       size_t num_growths, size_t K, size_t k,
                                       double rho_min,
                                       const PairWiseDrawer* drawer) const
  {
    auto comment = std::string{};
    comment = "Evaluating outlier resistance on dataset '";
    comment += dataset().name() + "' :\n\tpair 1-" + to_string(img_index + 1);
    comment += "\n\tfeatType = " + dataset().featType();
    comment += "\n\tsquaredEll = " + to_string(squared_ell);
    comment += "\n\tK = " + to_string(K);
    comment += "\n\trho_min = " + to_string(rho_min);
    comment + "_inlierThres_" + to_string(inlier_thres);
    print_stage(comment);

    // Get subset of matches.
    vector<size_t> inliers, outliers;
    get_inliers_and_outliers(inliers, outliers, M, H, inlier_thres);

    // We want to perform our analysis on this particular subset of matches of
    // interest.
    bool verbose = _debug && (drawer != nullptr);
    RegionGrowingAnalyzer analyzer(M, H, verbose);
    analyzer.set_inliers(inliers);

    // Grow multiple regions.
    cout << "Growing Regions... ";
    GrowthParams params(K, rho_min);
    GrowMultipleRegions growMultipleRegions(M, params, _debug ? 1 : 0);
    vector<Region> RR(growMultipleRegions(num_growths, &analyzer, drawer));
    cout << "Done!" << endl;

    // Compute the statistics.
    cout << "Computing stats... ";
    // Get found matches in a proper container.
    vector<size_t> all_matches;
    {
      Region all_R;
      for (size_t i = 0; i != RR.size(); ++i)
        for (Region::iterator j = RR[i].begin(); j != RR[i].end(); ++j)
          all_R.insert(*j);
      all_matches.reserve(all_R.size());
      for (Region::iterator i = all_R.begin(); i != all_R.end(); ++i)
        all_matches.push_back(*i);
    }
    analyzer.compute_positives_and_negatives(all_matches);

    // Save stats.
    cout << "Saving stats... ";
    string folder;
    folder = dataset().name() + "/outlier_resistance";
    folder = string_src_path(folder);
#pragma omp critical
    {
      mkdir(folder);
    }

    const string name(
        dataset().name() + "_" + to_string(1) + "_" + to_string(img_index + 1) +
        "_sqEll_" + to_string(squared_ell) + "_nReg_ " + to_string(num_growths) +
        "_K_" + to_string(K) + "_rhoMin_" + to_string(rho_min) + "_inlierThres_" +
        to_string(inlier_thres) + dataset().featType() + ".txt");

    bool success;
#pragma omp critical
    {
      success = analyzer.save_precision_recall_etc(string(folder + "/" + name));
    }
    if (!success)
    {
      cerr << "Could not save stats:\n" << string(folder + "/" + name) << endl;
      return false;
    }
    cout << "Done!" << endl;
    return true;
  }

} /* namespace DO::Sara */
