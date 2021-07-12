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

#include "Learn_P_d.hpp"

#include "LocalAffineConsistency.hpp"
#include "MatchNeighborhood.hpp"

#include <DO/Sara/FileSystem/FileSystem.hpp>
#include <DO/Sara/Graphics/Match/PairWiseDrawer.hpp>


using namespace std;


namespace DO::Sara {

  bool LearnPf::operator()(float inlier_thres, float squared_ell)
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

        // Compute initial matches $\mathcal{M}$.
        const auto M = compute_matches(X, Y, squared_ell);

        // Get inliers
        vector<size_t> inliers, outliers;
        const Matrix3f& H = dataset().H(j);
        get_inliers_and_outliers(inliers, outliers, M, H, inlier_thres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        get_key();

        for (size_t i = 0; i != inliers.size(); ++i)
        {
          const Match& m = M[inliers[i]];
          const OERegion& x = m.x();
          const OERegion& y = m.y();
          OERegion H_x = transform_oeregion(x, H);

          Ellipse H_Sx = ellipse_from_oeregion(H_x);
          Ellipse Sy = ellipse_from_oeregion(y);

          //if (_debug)
          //{
          //  drawer.draw_match(m, Green8);
          //  cout << H_Sx << endl;
          //}

          const auto polygonal_overlap =
              area(approximate_intersection(H_Sx, Sy, 36));
          const auto analytical_overlap = analytic_intersection_area(H_Sx, Sy);
          const auto angle_phi_ox = compute_orientation(m, H);

          // const auto theta_x = m.x().orientation;
          const auto theta_y = m.y().orientation;

          Vector2d Phi_ox = unit_vector2(angle_phi_ox);
          const auto oy = unit_vector2(double(theta_y));

          if (_debug)
          {
            // Verbose comment
            cout << "(Polygonal Approx.) Overlap ratio = " << polygonal_overlap
                 << endl;
            cout << "(Analytical Comp. ) Overlap ratio = " << analytical_overlap
                 << endl;
            cout << "Phi_theta_x ~ " << to_degree(angle_phi_ox) << " deg"
                 << endl;
            cout << "theta_y     = " << to_degree(theta_y) << " deg" << endl;
            cout << "Phi(ox) * oy = " << Phi_ox.dot(oy) << endl;
            cout << "|Phi_theta_x - theta_y| = "
                 << to_degree(abs(angle_phi_ox - theta_y)) << " deg" << endl;
            cout << endl;

            // Draw match
            drawer.draw_match(m);

            // Draw orientation.
            Phi_ox *= 20.;
            Vector2d p1, p2;
            Vector2d off(drawer.offF(1).cast<double>());
            p1 = H_Sx.center() + off;
            p2 = p1 + Phi_ox;
            draw_arrow(p1.x(), p1.y(), p2.x(), p2.y(), Blue8);

            // Draw transformed ellipse.
            H_Sx.center() += off;
            draw_ellipse(H_Sx, Blue8);
            fill_circle(H_Sx.center().x(), H_Sx.center().y(), 5, Blue8);
          }
        }
      }
      close_window_for_image_pair();
    }

    const auto folder = string_src_path(dataset().name() + "/P_d");
    mkdir(folder);

    return true;
  }

} /* namespace DO::Sara */
