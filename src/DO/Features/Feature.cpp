// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Features.hpp>
#include <DO/Graphics.hpp>
#include <Eigen/Eigen>

#define DRAW_ELLIPSE

namespace DO {

  //! Computes and return the scale given an input orientation
  float OERegion::radius(float angle) const
  {
    JacobiSVD<Matrix2f> svd(shape_matrix_, Eigen::ComputeFullU);
    const Vector2f radii(svd.singularValues().cwiseSqrt().cwiseInverse());
    const Matrix2f& U(svd.matrixU());
    //std::cout << theta/M_PI*180<< "degrees" << std::endl;
    Vector2f u(cos(angle), sin(angle));
    Vector2f e1(U.col(0));
    Vector2f e2(U.col(1));
    float x = radii(0)*e1.dot(u);
    float y = radii(1)*e2.dot(u);
    return sqrt(x*x+y*y);
  }

  Matrix3f OERegion::affinity() const
  {
    Matrix2f M = shapeMat();
    Matrix2f Q(Rotation2D<float>(orientation()).matrix());
    M = Q.transpose() * M * Q;
    Matrix2f R( Matrix2f(M.llt().matrixU()).inverse() );

    Matrix3f A;
    A.setZero();
    A.block(0,0,2,2) = Q*R;
    A.block(0,2,3,1) << center(), 1.f;
    return A;
  }

  void OERegion::drawOnScreen(const Color3ub& c, float z, const Point2f& off) const
  {
#ifdef DRAW_ELLIPSE
    // Solve characteristic equation to find eigenvalues
    JacobiSVD<Matrix2f> svd(shape_matrix_, ComputeFullU);
    const Vector2f& D = svd.singularValues();
    const Matrix2f& U = svd.matrixU();

    // Eigenvalues are l1 and l2.
    const Vector2f radii(D.cwiseSqrt().cwiseInverse());

    const float a = radii(0);
    const float b = radii(1);
    // The normalized eigenvectors/ellipsoid axes.
    Vector2f e1(U.col(0));
    Vector2f e2(U.col(1));

    float ellOrient = atan2(U(1,0), U(0,0));

    Matrix2f L = affinity().block(0,0,2,2);    
#endif

    Vector2f p1(z*(center()+off));
    float cross_offset=3.0f;
    Vector2f c1(p1-Vector2f(cross_offset,0.f));
    Vector2f c2(p1+Vector2f(cross_offset,0.f));
    Vector2f c3(p1-Vector2f(0.f,cross_offset));
    Vector2f c4(p1+Vector2f(0.f,cross_offset));
    drawLine(c1, c2, Black8, 5);
    drawLine(c3, c4, Black8, 5);
#ifdef DRAW_ELLIPSE
    //Vector2f p2(p1 + z*r*u);
    Vector2f p2(p1 + z*L*Vector2f(1.f,0.f));

    if (z*a > 1.f && z*b > 1.f && (p1-p2).squaredNorm() > 1.f)
    {
      drawLine(p1, p2, Black8, 5); // Contour.
      drawEllipse(p1, z*a, z*b, 180.f*ellOrient/float(M_PI), Black8, 5); // Contour.
      drawLine(p1, p2, c, 3); // Fill-in.
      drawEllipse(p1, z*a, z*b, 180.f*ellOrient/float(M_PI), c, 3); // Fill-in.
    }
#endif
    //fillCircle(p1, 1.f, c);
    drawLine(c1, c2, c, 3);
    drawLine(c3, c4, c, 3);
  }

} /* namespace DO */