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

#include "../GrowMultipleRegions.hpp"
#include "../MatchNeighborhood.hpp"
#include "EvaluateQualityOfLocalAffineApproximation.hpp"


using namespace std;


namespace DO::Sara {

  bool EvalQualityOfLocalAffApprox::operator()(float squared_ell,
                                               size_t numRegionGrowths,
                                               size_t K, double rho_min) const
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
    // float mikolajczykInlierThres = 1.5f;
    // Set of thresholds.
    vector<float> thres;
    thres.push_back(0.f);
    thres.push_back(1.5f);
    thres.push_back(5.f);
    thres.push_back(10.f);
    thres.push_back(20.f);
    thres.push_back(30.f);

    // ====================================================================== //
    // Let's go.
    for (int j = 1; j < 6; ++j)
    {
      PairWiseDrawer* drawer = 0;
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
        // Extract the subset of matches of interest.
        vector<IndexDist> M_sorted(sort_matches_by_reprojection_error(M, H));

        for (size_t lb = 0; lb != thres.size() - 1; ++lb)
        {
          int ub = lb + 1;
          bool success;
          success = run(M, M_sorted, H, j, squared_ell, thres[lb], thres[ub],
                             numRegionGrowths, K, rho_min, drawer);
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

  bool EvalQualityOfLocalAffApprox::run(const vector<Match>& M,
                                        const vector<IndexDist>& M_sorted,
                                        const Matrix3f& H, size_t img_index,
                                        float squared_ell, float lb, float ub,
                                        size_t num_growths, size_t K,
                                        double rho_min,
                                        const PairWiseDrawer* drawer) const
  {
    auto comment = std::string{};
    comment = dataset().name() + ":\n\tpair 1-" + to_string(img_index + 1);
    comment += "\n\tfeatType = " + dataset().featType();
    comment += "\n\tsquaredEll = " + to_string(squared_ell);
    comment += "\n\tK = " + to_string(K);
    comment += "\n\trho_min = " + to_string(rho_min);
    comment += "\n\tlb = " + to_string(lb);
    comment += "\n\tub = " + to_string(ub);
    print_stage(comment);

    // Get subset of matches.
    vector<size_t> I(get_matches(M_sorted, lb, ub));
    // We want to perform our analysis on this particular subset of matches of
    // interest.
    bool verbose = _debug && drawer;
    RegionGrowingAnalyzer analyzer(M, H, verbose);
    analyzer.set_subset_of_interest(I);

    // Grow multiple regions.
    cout << "Growing Regions... ";
    GrowthParams params(K, rho_min);
    GrowMultipleRegions growMultipleRegions(M, params, _debug ? 1 : 0);
    vector<Region> RR(growMultipleRegions(num_growths, &analyzer, drawer));
    cout << "Done!" << endl;

    // Compute the statistics.
    cout << "Computing stats... ";
    analyzer.compute_local_affine_consistency_statistics();
    cout << "Done!" << endl;

    // Save stats.
    cout << "Saving stats... ";
    string folder;
    folder = dataset().name() + "/Quality_Local_Aff";
    folder = string_src_path(folder);
#pragma omp critical
    {
      mkdir(folder);
    }

    const string name(
        dataset().name() + "_" + to_string(1) + "_" + to_string(img_index + 1) +
        "_sqEll_" + to_string(squared_ell) + "_nReg_ " + to_string(num_growths) +
        "_K_" + to_string(K) + "_rhoMin_" + to_string(rho_min) + "_lb_" +
        to_string(lb) + "_ub_" + to_string(ub) + dataset().featType() + ".txt");

    bool success;
#pragma omp critical
    {
      success =
          analyzer.save_local_affine_consistency_statistics(string(folder + "/" + name));
    }
    if (!success)
    {
      cerr << "Could not save stats:\n" << string(folder + "/" + name) << endl;
      return false;
    }
    cout << "Done!" << endl;
    return true;
  }

}  // namespace DO::Sara
