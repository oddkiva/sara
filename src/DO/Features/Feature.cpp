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

using namespace std;

namespace DO {

  std::ostream& InterestPoint::print(std::ostream& os) const
  {
    os << "Feature type:\t";
    switch (type())
    {
    case InterestPoint::DoG:
      os << "DoG" << endl;
      break;
    case InterestPoint::HarAff:
      os << "Harris-Affine" << endl;
      break;
    case InterestPoint::HesAff:
      os << "Hessian-Affine" << endl;
      break;
    case InterestPoint::MSER:
      os << "MSER" << endl;
      break;
    default:
      break;
    }
    os << "Position:\t" << coords().transpose() << endl;
    os << "Extremum type:\t" << extremum_type() << endl;
    os << "Extremum value:\t" << extremum_value() << endl;
    return os;
  }
  
  std::istream& InterestPoint::read(std::istream& in)
  { 
    return in >> x() >> y();
  }

  //! Computes and return the scale given an input orientation
  float OERegion::radius(float angle) const
  {
    JacobiSVD<Matrix2f> svd(_shape_matrix, Eigen::ComputeFullU);
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
    Matrix2f M = shape_matrix();
    Matrix2f Q(Rotation2D<float>(orientation()).matrix());
    M = Q.transpose() * M * Q;
    Matrix2f R( Matrix2f(M.llt().matrixU()).inverse() );

    Matrix3f A;
    A.setZero();
    A.block(0,0,2,2) = Q*R;
    A.block(0,2,3,1) << center(), 1.f;
    return A;
  }

  static inline float to_degree(float radian)
  {
     return radian/float(M_PI)*180.f;
  }

  ostream& OERegion::print(ostream& os) const
  {
    return InterestPoint::print(os) 
      << "shape matrix:\n" << shape_matrix() << endl
      << "orientation:\t" << to_degree(orientation()) << " degrees" << endl;
  }

  istream& OERegion::read(istream& in)
  {
    int feature_type;
    InterestPoint::read(in) 
      >> _shape_matrix
      >> _orientation
      >> feature_type;
    type() = static_cast<Type>(feature_type);
    return in;
  }

} /* namespace DO */