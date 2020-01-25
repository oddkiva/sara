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
#include "MatchNeighborhood.hpp"

using namespace std;

namespace DO::Sara {

  static Matrix2d jacobian(const Matrix3d& H, const Vector2d& x)
  {
    Matrix2d J;
    Vector3d xh; xh << x, 1.;
    double h1_xh = H.row(0)*xh;
    double h2_xh = H.row(1)*xh;
    double h3_xh = H.row(2)*xh;

    RowVector2d h1_t = H.row(0).block(0,0,2,1);
    RowVector2d h2_t = H.row(1).block(0,0,2,1);
    RowVector2d h3_t = H.row(2).block(0,0,2,1);

    J.row(0) = (h1_t*h3_xh -h1_xh*h3_t)/(h3_xh*h3_xh);
    J.row(1) = (h2_t*h3_xh -h2_xh*h3_t)/(h3_xh*h3_xh);

    return J;
  }

  static Matrix3d affinity(const Matrix3d& H, const Vector2d& x)
  {
    Vector3d H_xh;
    H_xh << x, 1.;
    H_xh = H*H_xh;
    H_xh /= H_xh(2);

    Vector2d Hx; Hx = H_xh.block(0,0,1,2);
    Matrix2d Jx; Jx = jacobian(H, x);

    Matrix3d A;
    A.setZero();
    A.block(0,0,2,2) = Jx; A.block(0,2,1,2) = Hx-Jx*x;
    A(2,2) = 1.;
    return A;
  }

  static Ellipse computeEllipse(const Match& m, const Matrix3f& Hf)
  {
    Ellipse L_ex;

    Matrix3d H = Hf.cast<double>();
    Matrix3d invH = H.inverse();

    Vector3d H_xh; H_xh << m.posX().cast<double>(), 1;
    H_xh = H*H_xh;
    H_xh /= H_xh(2);
    Vector2d H_x; H_x = H_xh.block(0,0,1,2);

    Matrix2d L, invL;
    L = jacobian(H, m.posX().cast<double>());
    invL = L.inverse();

    Matrix2d L_Sx = m.featX().shapeMat().cast<double>();
    L_Sx = invL.transpose()*L_Sx*invL;

    Matrix2d Sx = m.featX().shapeMat().cast<double>();
    /*Ellipse ex = fromShapeMat(Sx, m.posX().cast<double>());
    ex.drawOnScreen(Red8);*/

    L_ex = fromShapeMat(L_Sx, H_x);

    return L_ex;
  }

  static double computeOrientation(const Match& m, const Matrix3f& Hf)
  {
    double anglex = m.featX().orientation();
    Vector2d ox(cos(anglex), sin(anglex));

    double angley = m.featY().orientation();

    Matrix2d L(jacobian(Hf.cast<double>(), m.posX().cast<double>()));
    Vector2d L_ox(L*ox);
    L_ox.normalize();

    anglex = atan2(L_ox(1), L_ox(0));

    return anglex;
  }

  bool LearnPf::operator()(float inlierThres, float squaredEll)
  {
    for (int j = 1; j < 6; ++j)
    {
      // View the image pair.
      openWindowForImagePair(0, j);
      PairWiseDrawer drawer(dataset().image(0), dataset().image(j));
      drawer.setVizParams(1.0f, 1.0f, PairWiseDrawer::CatH);
      drawer.displayImages();
      {
        // Read the set of keypoints $\mathcal{X}$ in image 1.
        const vector<Keypoint>& X = dataset().keys(0);
        // Read the set of keypoints $\mathcal{Y}$ in image 2.
        const vector<Keypoint>& Y = dataset().keys(j);

        // Compute initial matches $\mathcal{M}$.
        vector<Match> M(computeMatches(X, Y, squaredEll));

        // Get inliers
        vector<size_t> inliers, outliers;
        const Matrix3f& H = dataset().H(j);
        getInliersAndOutliers(inliers, outliers, M, H, inlierThres);
        cout << "inliers.size() = " << inliers.size() << endl;
        cout << "outliers.size() = " << outliers.size() << endl;

        for (size_t i = 0; i != inliers.size(); ++i)
        {
          const Match& m = M[inliers[i]];

          if (debug)
          {
            drawer.drawMatch(m, Green8);
            cout << H_Sx << endl;
          }

          Ellipse H_Sx = computeEllipse(m, H);
          Ellipse Sy = fromShapeMat(m.featY().shapeMat().cast<double>(),
                                    m.posY().cast<double>());


          double polyApproxOverlap = approximateIntersectionUnionRatio(H_Sx, Sy);
          double analyticalOverlap = analyticInterUnionRatio(H_Sx, Sy);
          double orient = computeOrientation(m, H);
          double angle_x = m.featX().orientation();
          double angle_y = m.featY().orientation();
          Vector2d Phi_ox(unitVector2(orient)), oy(unitVector2(ay));          

          if (debug)
          {         
            // Verbose comment
            cout << "(Polygonal Approx.) Overlap ratio = " << appOverlap << endl;
            cout << "(Analytical Comp. ) Overlap ratio = " << overlap << endl;
            cout << "Phi_theta_x ~ " << toDegree(orient) << " deg" << endl;
            cout << "theta_y     = " << toDegree(ay) << " deg" << endl;
            cout << "Phi(ox)*oy = " << Phi_ox.dot(oy) << endl;
            cout << "|Phi_theta_x - theta_y| = " << toDegree(abs(orient-ay)) << " deg" << endl;
            cout << endl;
          
            // Draw match
            drawer.drawMatch(inliers[i]);
            // Draw orientation.
            Phi_ox *= 20.;
            Vector2d p1, p2;
            p1 = H_Sx.c() + off;
            p2 = p1 + Phi_ox;
            drawArrow(p1.x(), p1.y(), p2.x(), p2.y(), Blue8);
            // Draw transformed ellipse.
            H_Sx.c() += off;
            H_Sx.drawOnScreen(Blue8);
            fillCircle(H_Sx.c().x(), H_Sx.c().y(), 5, Blue8);
            getKey();
          }
        }
      }
      closeWindowForImagePair();
    }

    string folder(dataset().name()+"/P_f");
    folder = stringSrcPath(folder);
    createDirectory(folder);
    
    return true;
  }

} /* namespace DO::Sara */
