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

  bool Study_N_K_m::operator()(float inlierThres, float squaredEll, size_t K,
                               double squaredRhoMin)
  {
    // Store stats.
    vector<Stat> stat_N_Ks, stat_hatN_Ks, stat_diffs;
    vector<Stat> stat_N_Ks_inliers, stat_hatN_Ks_inliers, stat_diffs_inliers;
    vector<Stat> stat_N_Ks_outliers, stat_hatN_Ks_outliers, stat_diffs_outliers;

    for (int j = 1; j < 6; ++j)
    {
      // View the image pair.
      openWindowForImagePair(0, j);
      PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
      drawer.setVizParams(1.0f, 1.0f, PairWiseDrawer::CatH);
      drawer.displayImages();
      {
        /*for (size_t i = 0; i != M.size(); ++i)
          drawer.drawMatch(M[i]);*/

        // Read the set of keypoints $\mathcal{X}$ in image 1.
        const Set<OERegion, RealDescriptor>& X = dataset().keys(0);
        // Read the set of keypoints $\mathcal{Y}$ in image 2.
        const Set<OERegion, RealDescriptor>& Y = dataset().keys(j);
        // Compute initial matches.
        vector<Match> M(computeMatches(X, Y, squaredEll));

        // ComputeMatchNeighborhood computeNeighborhood(filteredM, 1e3, &drawer,
        // true);
        ComputeN_K computeN_K(M, 1e3);

        // Get inliers and outliers.
        vector<size_t> inliers, outliers;
        getInliersAndOutliers(inliers, outliers, M, dataset().H(j),
                              inlierThres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;

        // Compute neighborhoods.
        vector<vector<size_t>> N_K(computeN_K(K, squaredRhoMin));
        vector<vector<size_t>> hatN_K(computeHatN_K(N_K));

        // General stats.
        Stat stat_N_K, stat_hatN_K, stat_diff;
        getStat(stat_N_K, stat_hatN_K, stat_diff, N_K, hatN_K);
        stat_N_Ks.push_back(stat_N_K);
        stat_hatN_Ks.push_back(stat_hatN_K);
        stat_diffs.push_back(stat_diff);
        // Stats with inliers only.
        Stat inlier_stat_N_K, inlier_stat_hatN_K, inlier_stat_diff;
        getStat(inlier_stat_N_K, inlier_stat_hatN_K, inlier_stat_diff, inliers,
                N_K, hatN_K);
        stat_N_Ks_inliers.push_back(inlier_stat_N_K);
        stat_hatN_Ks_inliers.push_back(inlier_stat_hatN_K);
        stat_diffs_inliers.push_back(inlier_stat_diff);
        // Stats with outliers only.
        Stat outlier_stat_N_K, outlier_stat_hatN_K, outlier_stat_diff;
        getStat(outlier_stat_N_K, outlier_stat_hatN_K, outlier_stat_diff,
                outliers, N_K, hatN_K);
        stat_N_Ks_outliers.push_back(outlier_stat_N_K);
        stat_hatN_Ks_outliers.push_back(outlier_stat_hatN_K);
        stat_diffs_outliers.push_back(outlier_stat_diff);
      }
      closeWindowForImagePair();
    }

    string folder(dataset().name());
    folder = stringSrcPath(folder);
    string folder_inlier = folder + "_inlier";
    string folder_outlier = folder + "_outlier";
    createDirectory(folder);
    createDirectory(folder_inlier);
    createDirectory(folder_outlier);

    const string name("squaredEll_" + toString(squaredEll) + "_K_" +
                      toString(K) + "_squaredRhoMin_" +
                      toString(squaredRhoMin) + dataset().featType() + ".txt");
    // General stats
    if (!saveStats(folder + "/" + name, stat_N_Ks, stat_hatN_Ks, stat_diffs))
    {
      cerr << "Could not save stats:\n" << string(folder + name) << endl;
      return false;
    }
    cout << "Saved stats:\n" << string(folder + "/" + name) << endl;
    // Inlier stats
    if (!saveStats(folder_inlier + "/" + name, stat_N_Ks_inliers,
                   stat_hatN_Ks_inliers, stat_diffs_inliers))
    {
      cerr << "Could not save stats:\n"
           << string(folder_inlier + "/" + name) << endl;
      return false;
    }
    cout << "Saved stats:\n" << string(folder_inlier + "/" + name) << endl;
    // Outlier stats
    if (!saveStats(folder_outlier + "/" + name, stat_N_Ks_outliers,
                   stat_hatN_Ks_outliers, stat_diffs_outliers))
    {
      cerr << "Could not save stats:\n" << string(folder + name) << endl;
      return false;
    }
    cout << "Saved stats:\n" << string(folder_outlier + "/" + name) << endl;

    return true;
  }

  void Study_N_K_m::getStat(Stat& stat_N_K, Stat& stat_hatN_K, Stat& stat_diff,
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

    stat_diff.computeStats(diff_size);
    stat_N_K.computeStats(size_N_K);
    stat_hatN_K.computeStats(size_hatN_K);
  }

  void Study_N_K_m::checkNeighborhood(const vector<vector<size_t>>& N_K,
                                      const vector<Match>& M,
                                      const PairWiseDrawer& drawer)
  {
    drawer.displayImages();
    for (size_t i = 0; i != M.size(); ++i)
    {
      cout << "N_K[" << i << "].size() = " << N_K[i].size() << "\n ";
      for (size_t j = 0; j != N_K[i].size(); ++j)
        cout << N_K[i][j] << " ";
      cout << endl;

      drawer.displayImages();
      for (size_t j = 0; j != N_K[i].size(); ++j)
      {
        size_t indj = N_K[i][j];
        drawer.drawMatch(M[indj]);
      }
      drawer.drawMatch(M[i], Red8);
      string name = "N_K_" + toString(i) + ".png";
      saveScreen(activeWindow(), stringSrcPath(name));
      getKey();
    }
  }

  void Study_N_K_m::getStat(Stat& stat_N_K, Stat& stat_hatN_K, Stat& stat_diff,
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
      N_K[i].reserve(1e3);
      hatN_K[i].reserve(1e3);
    }

    for (size_t i = 0; i != indices.size(); ++i)
    {
      N_K[i] = all_N_K[indices[i]];
      hatN_K[i] = all_hatN_K[indices[i]];
    }

    getStat(stat_N_K, stat_hatN_K, stat_diff, N_K, hatN_K);
  }

  bool Study_N_K_m::saveStats(const string& name, const vector<Stat>& stat_N_Ks,
                              const vector<Stat>& stat_hatN_Ks,
                              const vector<Stat>& stat_diffs)
  {
    ofstream out(name.c_str());
    if (!out.is_open())
      return false;

    out << "Statistics: N_K" << endl;
    writeStats(out, stat_N_Ks);
    out << "Statistics: stat_hatN_Ks" << endl;
    writeStats(out, stat_hatN_Ks);
    out << "Statistics: stat_diffs" << endl;
    writeStats(out, stat_diffs);
    out.close();

    return true;
  }

}  // namespace DO::Sara
