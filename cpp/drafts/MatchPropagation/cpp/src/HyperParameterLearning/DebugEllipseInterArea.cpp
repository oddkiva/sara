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

using namespace std;

namespace DO::Sara {

  bool DebugEllipseInterArea::operator()(float inlierThres, float squaredEll)
  {
    vector<Stat> stat_overlaps, stat_angles;

    for (int j = 1; j < 6; ++j)
    {
      // View the image pair.
      openWindowForImagePair(0, j);
      PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
      drawer.setVizParams(1.0f, 1.0f, PairWiseDrawer::CatH);
      drawer.displayImages();
      {
        // Read the set of keypoints $\mathcal{X}$ in image 1.
        const Set<OERegion, RealDescriptor>& X = dataset().keys(0);
        // Read the set of keypoints $\mathcal{Y}$ in image 2.
        const Set<OERegion, RealDescriptor>& Y = dataset().keys(j);

        // Compute initial matches $\mathcal{M}$.
        vector<Match> M(computeMatches(X, Y, squaredEll));

        // Get inliers
        vector<size_t> inliers, outliers;
        const Matrix3f& H = dataset().H(j);
        getInliersAndOutliers(inliers, outliers, M, H, inlierThres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;
        getKey();


        // Store overlaps
        vector<double> overlaps, angles;
        int num_error = 0;

        for (size_t i = 0; i != inliers.size(); ++i)
        {
          const Match& m = M[inliers[i]];
          const OERegion& x = m.x();
          const OERegion& y = m.y();
          OERegion H_x = transformOERegion(x, H);

          Ellipse H_Sx = ellipseFromOERegion(H_x);
          Ellipse Sy = ellipseFromOERegion(y);

          float diff_center = sqrt((H_x.center() - y.center()).squaredNorm());
          Matrix2f diff_shape_mat = H_x.shapeMat() - y.shapeMat();
          float rel_diff_shape_mat =
              diff_shape_mat.squaredNorm() / y.shapeMat().squaredNorm();

          /*if (debug_)
          {
            drawer.drawMatch(m, Green8);
            cout << H_Sx << endl;
          }*/

          Point2d inter[4];
          int num_inter;
          getEllipseIntersections(inter, num_inter, H_Sx, Sy);

          double polyApproxOverlap =
              approximateIntersectionUnionRatio(H_Sx, Sy);
          double analyticalOverlap = analyticInterUnionRatio(H_Sx, Sy);
          double angle_phi_ox = computeOrientation(m, H);
          // double angle_x = m.featX().orientation();
          double angle_y = m.y().orientation();
          double error =
              (polyApproxOverlap - analyticalOverlap) / polyApproxOverlap;

          if (debug_ && (error > 0.2))
          {
            ++num_error;
            cout << "M[" << i << "] = " << endl;
            cout << "diff_center = " << diff_center << endl;
            cout << "diff_shape_mat = \n" << diff_shape_mat << endl;
            cout << "rel_diff_shape_mat = \n" << rel_diff_shape_mat << endl;
            cout << "num_inter = " << num_inter << endl;

            drawer.displayImages();
            checkReprojectedEllipse(m, drawer, Sy, H_Sx, polyApproxOverlap,
                                    analyticalOverlap, angle_phi_ox, angle_y,
                                    error);
            // drawer.drawKeypoint(1, H_x, Yellow8);
            getKey();
          }

          overlaps.push_back(1 - polyApproxOverlap);
          angles.push_back(toDegree(abs(angle_phi_ox - angle_y)));
        }
        cout << "num error (ellipse) = " << num_error << endl;
        cout << "error rate = " << double(num_error) / inliers.size() << endl;

        Stat stat_overlap, stat_angle;
        stat_overlap.computeStats(overlaps);
        stat_angle.computeStats(angles);

        stat_overlaps.push_back(stat_overlap);
        stat_angles.push_back(stat_angle);
      }
      closeWindowForImagePair();
    }

    string folder(dataset().name() + "/P_f");
    folder = stringSrcPath(folder);
    createDirectory(folder);

    const string name("inlierThres_" + toString(inlierThres) + "_squaredEll_" +
                      toString(squaredEll) + dataset().featType() + ".txt");

    if (!saveStats(folder + "/" + name, stat_overlaps, stat_angles))
    {
      cerr << "Could not save stats:\n" << string(folder + "/" + name) << endl;
      return false;
    }
    return true;
  }

  void DebugEllipseInterArea::checkReprojectedEllipse(
      const Match& m, const PairWiseDrawer& drawer, Ellipse& y, Ellipse& H_Sx,
      double polyApproxOverlap, double analyticalOverlap, double angle_phi_ox,
      double angle_y, double error) const
  {
    Vector2d Phi_ox(unitVector2(angle_phi_ox));
    Vector2d oy(unitVector2(angle_y));

    // Verbose comment
    cout << "(Polygonal Approx.) Overlap ratio = " << polyApproxOverlap << endl;
    cout << "(Analytical Comp. ) Overlap ratio = " << analyticalOverlap << endl;
    cout << "angle_phi_ox = " << toDegree(angle_phi_ox) << " deg" << endl;
    cout << "angle_y     = " << toDegree(angle_y) << " deg" << endl;
    cout << "Phi(ox)*oy = " << Phi_ox.dot(oy) << endl;
    cout << "|Phi_theta_x - theta_y| = "
         << toDegree(abs(angle_phi_ox - angle_y)) << " deg" << endl;
    cout << endl;

    /*if (error > 0.2)
    {*/
    cout << "WARNING: analytical computation grossly inaccurate" << endl;
    cout << "error = " << error << endl << endl;
    /*}*/

    // Draw match
    // drawer.drawMatch(m);

    // Draw orientation.
    Phi_ox *= 20.;
    Vector2d p1, p2;
    Vector2d off(drawer.offF(1).cast<double>());
    p1 = H_Sx.c() + off;
    p2 = p1 + Phi_ox;
    drawArrow(p1.x(), p1.y(), p2.x(), p2.y(), Blue8);
    // Draw transformed ellipse.
    H_Sx.c() += off;
    H_Sx.drawOnScreen(Blue8);
    fillCircle(H_Sx.c().x(), H_Sx.c().y(), 5, Blue8);

    y.c() += off;
    y.drawOnScreen(Red8);
    fillCircle(y.c().x(), y.c().y(), 5, Blue8);
  }

  bool DebugEllipseInterArea::saveStats(const string& name,
                                        const vector<Stat>& stat_overlaps,
                                        const vector<Stat>& stat_angles) const
  {
    ofstream out(name.c_str());
    if (!out.is_open())
      return false;

    out << "Statistics: overlaps" << endl;
    writeStats(out, stat_overlaps);
    out << "Statistics: angles" << endl;
    writeStats(out, stat_angles);
    out.close();

    return true;
  }

} /* namespace DO::Sara */
