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

#include "Learn_P_f.hpp"

#include "../LocalAffineConsistency.hpp"
#include "../MatchNeighborhood.hpp"

#include <DO/Sara/FileSystem/FileSystem.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif


using namespace std;


namespace DO::Sara {

  bool LearnPf::operator()(float squared_ell) const
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
    thres.push_back(40.f);
    thres.push_back(50.f);
    thres.push_back(100.f);
    thres.push_back(200.f);

    // Array of stats.
    vector<vector<Statistics>> stat_overlaps(thres.size() - 1),
        stat_angles(thres.size() - 1);

    // Let's go.
    for (int j = 1; j < 6; ++j)
    {
      // View the image pair.
      if (display_)
      {
        open_window_for_image_pair(0, j);
        PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
        drawer.set_viz_params(1.0f, 1.0f, PairWiseDrawer::CatH);
        drawer.display_images();
      }

      // Read the set of keypoints $\mathcal{X}$ in image 1.
      const auto& X = dataset().keys(0);
      // Read the set of keypoints $\mathcal{Y}$ in image 2.
      const auto& Y = dataset().keys(j);

      // Compute initial matches $\mathcal{M}$.
      const auto M = compute_matches(X, Y, squared_ell);

      // Get inliers
      vector<size_t> inliers, outliers;
      const Matrix3f& H = dataset().H(j);
      get_inliers_and_outliers(inliers, outliers, M, H, thres[1]);
      cout << "inliers.size() = " << inliers.size() << endl;
      cout << "outliers.size() = " << outliers.size() << endl;
      get_key();

      // Extract the subset of matches of interest.
      const auto M_sorted = sort_matches_by_reprojection_error(M, H);

      for (size_t lb = 0; lb != thres.size() - 1; ++lb)
      {
        const auto ub = lb + 1;
        Statistics stat_overlap, stat_angle;
        run(stat_overlap, stat_angle, M, M_sorted, H, thres[lb],
            thres[ub] /*, &drawer*/);
        stat_overlaps[lb].push_back(stat_overlap);
        stat_angles[lb].push_back(stat_angle);
      }

      if (display_)
        close_window_for_image_pair();
    }

    // ====================================================================== //
    // Save stats.
    string folder;
    folder = approx_ell_inter_area_ ? dataset().name() + "/P_f_approx"
                                    : dataset().name() + "/P_f";
    folder = string_src_path(folder);
#pragma omp critical
    {
      mkdir(folder);
    }

    for (size_t lb = 0; lb != thres.size() - 1; ++lb)
    {
      size_t ub = lb + 1;
      const string name(dataset().name() + "_lb_" + to_string(thres[lb]) +
                        "_ub_" + to_string(thres[ub]) + "_squaredEll_" +
                        to_string(squared_ell) + dataset().feature_type() + ".txt");

      bool success;
#pragma omp critical
      {
        success = save_statistics(folder + "/" + name, stat_overlaps[lb],
                                  stat_angles[lb]);
      }

      if (!success)
      {
        cerr << "Could not save stats:\n"
             << string(folder + "/" + name) << endl;
        return false;
      }
    }
    return true;
  }

  bool LearnPf::save_statistics(const string& name,
                                const vector<Statistics>& stat_overlaps,
                                const vector<Statistics>& stat_angles) const
  {
    ofstream out(name.c_str());
    if (!out.is_open())
      return false;

    out << "Statistics: overlaps" << endl;
    write_statistics(out, stat_overlaps);
    out << "Statistics: angles" << endl;
    write_statistics(out, stat_angles);
    out.close();

    return true;
  }

  void LearnPf::run(Statistics& stat_overlap, Statistics& stat_angle,
                    const vector<Match>& M, const vector<IndexDist>& M_sorted,
                    const Matrix3f& H, float lb, float ub,
                    PairWiseDrawer* drawer) const
  {
    // Get subset of matches.
    const auto I = get_matches(M_sorted, lb, ub);

    // Store overlaps.
    vector<double> overlaps(I.size()), angles(I.size());

    // Compute stuff for statistics.
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(I.size()); ++i)
    {
      const Match& m = M[I[i]];
      const OERegion& x = m.x();
      const OERegion& y = m.y();
      OERegion H_x = transform_oeregion(x, H);

      float dist;
      double angle_diff_radian, overlap_ratio;
      compare_oeregions(dist, angle_diff_radian, overlap_ratio, H_x, y,
                        approx_ell_inter_area_);

      if (debug_)
      {
        cout << "dist = " << dist << endl;
        cout << "(Analytical Comp. ) Overlap ratio = " << overlap_ratio << endl;
        cout << "angle_H_ox = " << to_degree(H_x.orientation) << " deg" << endl;
        cout << "angle_y     = " << to_degree(y.orientation) << " deg" << endl;
        cout << "|angle_H_ox - angle_y| = " << angle_diff_radian << " deg"
             << endl
             << endl;
      }

      // View the image pair.
      drawer->draw_feature(1, H_x, Blue8);
      drawer->draw_feature(1, y, Red8);

      overlaps[i] = 1 - overlap_ratio;
      angles[i] = to_degree(angle_diff_radian);
    }

    // Compute the stats.
    stat_overlap.compute_statistics(overlaps);
    stat_angle.compute_statistics(angles);
  }

} /* namespace DO::Sara */
