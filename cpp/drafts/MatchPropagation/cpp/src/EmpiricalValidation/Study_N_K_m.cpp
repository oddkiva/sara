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

#include "Study_N_K_m.hpp"

#include "../MatchNeighborhood.hpp"


using namespace std;

namespace DO::Sara {

  bool Study_N_K_m::operator()(float inlier_thres, float squared_ell, size_t K,
                               double squaredRhoMin)
  {
    // Store stats.
    vector<Statistics> stat_N_Ks, stat_hatN_Ks, stat_diffs;
    vector<Statistics> stat_N_Ks_inliers, stat_hatN_Ks_inliers,
        stat_diffs_inliers;
    vector<Statistics> stat_N_Ks_outliers, stat_hatN_Ks_outliers,
        stat_diffs_outliers;

    for (int j = 1; j < 6; ++j)
    {
      // View the image pair.
      open_window_for_image_pair(0, j);
      PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
      drawer.set_viz_params(1.0f, 1.0f, PairWiseDrawer::CatH);
      drawer.display_images();
      {
        /*for (size_t i = 0; i != M.size(); ++i)
          drawer.draw_match(M[i]);*/

        // Read the set of keypoints $\mathcal{X}$ in image 1.
        const KeypointList<OERegion, float>& X = dataset().keys(0);
        // Read the set of keypoints $\mathcal{Y}$ in image 2.
        const KeypointList<OERegion, float>& Y = dataset().keys(j);
        // Compute initial matches.
        vector<Match> M(compute_matches(X, Y, squared_ell));

        // ComputeMatchNeighborhood computeNeighborhood(filteredM, 1e3, &drawer,
        // true);
        NearestMatchNeighborhoodComputer computeN_K(M, std::size_t(1e3));

        // Get inliers and outliers.
        vector<size_t> inliers, outliers;
        get_inliers_and_outliers(inliers, outliers, M, dataset().H(j),
                                 inlier_thres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;

        // Compute neighborhoods.
        const auto N_K = computeN_K(K, squaredRhoMin);
        const auto hatN_K = compute_hat_N_K(N_K);

        // General stats.
        Statistics stat_N_K, stat_hatN_K, stat_diff;
        compute_statistics(stat_N_K, stat_hatN_K, stat_diff, N_K, hatN_K);
        stat_N_Ks.push_back(stat_N_K);
        stat_hatN_Ks.push_back(stat_hatN_K);
        stat_diffs.push_back(stat_diff);

        // Stats with inliers only.
        Statistics inlier_stat_N_K, inlier_stat_hatN_K, inlier_stat_diff;
        compute_statistics(inlier_stat_N_K, inlier_stat_hatN_K,
                           inlier_stat_diff, inliers, N_K, hatN_K);
        stat_N_Ks_inliers.push_back(inlier_stat_N_K);
        stat_hatN_Ks_inliers.push_back(inlier_stat_hatN_K);
        stat_diffs_inliers.push_back(inlier_stat_diff);

        // Stats with outliers only.
        Statistics outlier_stat_N_K, outlier_stat_hatN_K, outlier_stat_diff;
        compute_statistics(outlier_stat_N_K, outlier_stat_hatN_K,
                           outlier_stat_diff, outliers, N_K, hatN_K);
        stat_N_Ks_outliers.push_back(outlier_stat_N_K);
        stat_hatN_Ks_outliers.push_back(outlier_stat_hatN_K);
        stat_diffs_outliers.push_back(outlier_stat_diff);
      }

      close_window_for_image_pair();
    }

    const auto folder = string_src_path(dataset().name());
    const auto folder_inlier = folder + "_inlier";
    const auto folder_outlier = folder + "_outlier";

    mkdir(folder);
    mkdir(folder_inlier);
    mkdir(folder_outlier);

    const string name("squaredEll_" + to_string(squared_ell) + "_K_" +
                      to_string(K) + "_squaredRhoMin_" +
                      to_string(squaredRhoMin) + dataset().featType() + ".txt");

    // General stats.
    if (!compute_statistics(folder + "/" + name, stat_N_Ks, stat_hatN_Ks,
                            stat_diffs))
    {
      cerr << "Could not save stats:\n" << string(folder + name) << endl;
      return false;
    }
    cout << "Saved stats:\n" << string(folder + "/" + name) << endl;

    // Inlier stats.
    if (!compute_statistics(folder_inlier + "/" + name, stat_N_Ks_inliers,
                            stat_hatN_Ks_inliers, stat_diffs_inliers))
    {
      cerr << "Could not save stats:\n"
           << string(folder_inlier + "/" + name) << endl;
      return false;
    }
    cout << "Saved stats:\n" << string(folder_inlier + "/" + name) << endl;

    // Outlier stats.
    if (!compute_statistics(folder_outlier + "/" + name, stat_N_Ks_outliers,
                            stat_hatN_Ks_outliers, stat_diffs_outliers))
    {
      cerr << "Could not save stats:\n" << string(folder + name) << endl;
      return false;
    }
    cout << "Saved stats:\n" << string(folder_outlier + "/" + name) << endl;

    return true;
  }

  void Study_N_K_m::compute_statistics(Statistics& stat_N_K,
                                       Statistics& stat_hatN_K,
                                       Statistics& stat_diff,
                                       const vector<vector<size_t>>& N_K,
                                       const vector<vector<size_t>>& hatN_K)
  {
    vector<int> diff_size(N_K.size());
    vector<int> size_N_K(N_K.size());
    vector<int> size_hatN_K(N_K.size());
    for (size_t i = 0; i != N_K.size(); ++i)
    {
      size_N_K[i] = N_K[i].size();
      size_hatN_K[i] = hatN_K[i].size();
      diff_size[i] = size_hatN_K[i] - size_N_K[i];
    }

    stat_diff.compute_statistics(diff_size);
    stat_N_K.compute_statistics(size_N_K);
    stat_hatN_K.compute_statistics(size_hatN_K);
  }

  void Study_N_K_m::check_neighborhood(const vector<vector<size_t>>& N_K,
                                       const vector<Match>& M,
                                       const PairWiseDrawer& drawer)
  {
    drawer.display_images();

    for (size_t i = 0; i != M.size(); ++i)
    {
      cout << "N_K[" << i << "].size() = " << N_K[i].size() << "\n ";
      for (size_t j = 0; j != N_K[i].size(); ++j)
        cout << N_K[i][j] << " ";
      cout << endl;

      drawer.display_images();
      for (size_t j = 0; j != N_K[i].size(); ++j)
      {
        size_t indj = N_K[i][j];
        drawer.draw_match(M[indj]);
      }
      drawer.draw_match(M[i], Red8);

      const auto name = "N_K_" + to_string(i) + ".png";
      save_screen(active_window(), string_src_path(name));
      get_key();
    }
  }

  void Study_N_K_m::compute_statistics(Statistics& stat_N_K,
                                       Statistics& stat_hatN_K,
                                       Statistics& stat_diff,
                                       const vector<size_t>& indices,
                                       const vector<vector<size_t>>& all_N_K,
                                       const vector<vector<size_t>>& all_hatN_K)
  {
    vector<vector<size_t>> N_K;
    vector<vector<size_t>> hatN_K;

    N_K.resize(indices.size());
    hatN_K.resize(indices.size());

    for (size_t i = 0; i != indices.size(); ++i)
    {
      N_K[i].reserve(std::size_t(1e3));
      hatN_K[i].reserve(std::size_t(1e3));
    }

    for (size_t i = 0; i != indices.size(); ++i)
    {
      N_K[i] = all_N_K[indices[i]];
      hatN_K[i] = all_hatN_K[indices[i]];
    }

    compute_statistics(stat_N_K, stat_hatN_K, stat_diff, N_K, hatN_K);
  }

  bool Study_N_K_m::compute_statistics(const string& name,
                                       const vector<Statistics>& stat_N_Ks,
                                       const vector<Statistics>& stat_hatN_Ks,
                                       const vector<Statistics>& stat_diffs)
  {
    ofstream out(name.c_str());
    if (!out.is_open())
      return false;

    out << "Statistics: N_K" << endl;
    write_statistics(out, stat_N_Ks);
    out << "Statistics: stat_hatN_Ks" << endl;
    write_statistics(out, stat_hatN_Ks);
    out << "Statistics: stat_diffs" << endl;
    write_statistics(out, stat_diffs);
    out.close();

    return true;
  }

}  // namespace DO::Sara
