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

#include "DebugEllipseInterArea.hpp"

#include "../LocalAffineConsistency.hpp"
#include "Statistics.hpp"


using namespace std;

namespace DO::Sara {

  bool DebugEllipseInterArea::operator()(float inlier_thres, float squared_ell)
  {
    vector<Statistics> stat_overlaps, stat_angles;

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

        // Compute initial matches $\mathcal{M}$.
        const auto M = compute_matches(X, Y, squared_ell);

        // Get inliers
        vector<size_t> inliers, outliers;
        const Matrix3f& H = dataset().H(j);
        get_inliers_and_outliers(inliers, outliers, M, H, inlier_thres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        get_key();

        // Store overlaps
        vector<double> overlaps, angles;
        int num_error = 0;

        for (size_t i = 0; i != inliers.size(); ++i)
        {
          const Match& m = M[inliers[i]];
          const OERegion& x = m.x();
          const OERegion& y = m.y();
          OERegion H_x = transform_oeregion(x, H);

          Ellipse H_Sx = ellipse_from_oeregion(H_x);
          Ellipse Sy = ellipse_from_oeregion(y);

          float diff_center = sqrt((H_x.center() - y.center()).squaredNorm());
          Matrix2f diff_shape_mat = H_x.shape_matrix - y.shape_matrix;
          float rel_diff_shape_mat =
              diff_shape_mat.squaredNorm() / y.shape_matrix.squaredNorm();

          /*if (_debug)
          {
            drawer.draw_match(m, Green8);
            cout << H_Sx << endl;
          }*/

          const auto inter = compute_intersection_points(H_Sx, Sy);

          const auto polygonal_overlap =
              approximate_jaccard_similarity(H_Sx, Sy, 36);
          const auto analytical_overlap = analytic_jaccard_similarity(H_Sx, Sy);

          const auto angle_phi_ox = compute_orientation(m, H);
          const auto angle_y = m.y().orientation;
          const auto error =
              (polygonal_overlap - analytical_overlap) / polygonal_overlap;

          if (_debug && (error > 0.2))
          {
            ++num_error;
            cout << "M[" << i << "] = " << endl;
            cout << "diff_center = " << diff_center << endl;
            cout << "diff_shape_mat = \n" << diff_shape_mat << endl;
            cout << "rel_diff_shape_mat = \n" << rel_diff_shape_mat << endl;
            cout << "num_inter = " << inter.size() << endl;

            drawer.display_images();
            check_reprojected_ellipse(m, drawer, Sy, H_Sx, polygonal_overlap,
                                      analytical_overlap, angle_phi_ox, angle_y,
                                      error);
            get_key();
          }

          overlaps.push_back(1 - polygonal_overlap);
          angles.push_back(to_degree(abs(angle_phi_ox - angle_y)));
        }
        cout << "num error (ellipse) = " << num_error << endl;
        cout << "error rate = " << double(num_error) / inliers.size() << endl;

        Statistics stat_overlap, stat_angle;
        stat_overlap.compute_statistics(overlaps);
        stat_angle.compute_statistics(angles);

        stat_overlaps.push_back(stat_overlap);
        stat_angles.push_back(stat_angle);
      }
      close_window_for_image_pair();
    }

    const auto folder = string_src_path(dataset().name() + "/P_f");
    mkdir(folder);

    const string name("inlierThres_" + to_string(inlier_thres) +
                      "_squaredEll_" + to_string(squared_ell) +
                      dataset().feature_type() + ".txt");

    if (!save_statistics(folder + "/" + name, stat_overlaps, stat_angles))
    {
      cerr << "Could not save stats:\n" << string(folder + "/" + name) << endl;
      return false;
    }
    return true;
  }

  void DebugEllipseInterArea::check_reprojected_ellipse(
      const Match& /* m */,                                 //
      const PairWiseDrawer& drawer,                         //
      Ellipse& y, Ellipse& H_Sx,                            //
      double polygonal_overlap, double analytical_overlap,  //
      double angle_phi_ox, double angle_y,                  //
      double error) const
  {
    Vector2d Phi_ox(unit_vector2(angle_phi_ox));
    Vector2d oy(unit_vector2(angle_y));

    // Verbose comment
    cout << "(Polygonal Approx.) Overlap ratio = " << polygonal_overlap << endl;
    cout << "(Analytical Comp. ) Overlap ratio = " << analytical_overlap
         << endl;
    cout << "angle_phi_ox = " << to_degree(angle_phi_ox) << " deg" << endl;
    cout << "angle_y     = " << to_degree(angle_y) << " deg" << endl;
    cout << "Phi(ox)*oy = " << Phi_ox.dot(oy) << endl;
    cout << "|Phi_theta_x - theta_y| = "
         << to_degree(abs(angle_phi_ox - angle_y)) << " deg" << endl;
    cout << endl;

    /*if (error > 0.2)
    {*/
    cout << "WARNING: analytical computation grossly inaccurate" << endl;
    cout << "error = " << error << endl << endl;
    /*}*/

    // Draw match
    // drawer.draw_match(m);

    // Draw orientation.
    Phi_ox *= 20.;
    Vector2d p1, p2;
    Vector2d off(drawer.offset(1).cast<double>());
    p1 = H_Sx.center() + off;
    p2 = p1 + Phi_ox;
    draw_arrow(p1.x(), p1.y(), p2.x(), p2.y(), Blue8);
    // Draw transformed ellipse.
    H_Sx.center() += off;
    draw_ellipse(H_Sx, Blue8);
    fill_circle(H_Sx.center().x(), H_Sx.center().y(), 5, Blue8);

    y.center() += off;
    draw_ellipse(y, Red8);
    fill_circle(y.center().x(), y.center().y(), 5, Blue8);
  }

  bool DebugEllipseInterArea::save_statistics(
      const string& name, const vector<Statistics>& stat_overlaps,
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

} /* namespace DO::Sara */
