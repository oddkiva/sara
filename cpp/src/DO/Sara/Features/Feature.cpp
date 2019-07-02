// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Features.hpp>


using namespace std;


namespace DO { namespace Sara {

  template <typename Enumeration>
  auto as_integer(Enumeration const value) ->
      typename std::underlying_type<Enumeration>::type
  {
    return static_cast<typename std::underlying_type<Enumeration>::type>(value);
  }

  //! Computes and return the scale given an input orientation
  float OERegion::radius(float angle) const
  {
    JacobiSVD<Matrix2f> svd(shape_matrix, Eigen::ComputeFullU);
    const Vector2f radii(svd.singularValues().cwiseSqrt().cwiseInverse());
    const Matrix2f& U(svd.matrixU());
    Vector2f u{cos(angle), sin(angle)};
    auto e1 = U.col(0);
    auto e2 = U.col(1);
    auto x = radii(0) * e1.dot(u);
    auto y = radii(1) * e2.dot(u);
    return sqrt(x * x + y * y);
  }

  Matrix3f OERegion::affinity() const
  {
    Matrix2f M = shape_matrix;
    auto Q = Rotation2D<float>(orientation).matrix();
    M = Q.transpose() * M * Q;
    Matrix2f R = Matrix2f{M.llt().matrixU()}.inverse();

    Matrix3f A;
    A.setZero();
    A.block(0, 0, 2, 2) = Q * R;
    A.block(0, 2, 3, 1) << center(), 1.f;
    return A;
  }

  static inline float to_degree(float radian)
  {
     return radian / float(M_PI) * 180.f;
  }

  ostream& operator<<(ostream& os, const OERegion& f)
  {
    os << "Feature type:\t";
    switch (f.type)
    {
    case OERegion::Type::DoG:
      os << "DoG" << endl;
      break;
    case OERegion::Type::HarAff:
      os << "Harris-Affine" << endl;
      break;
    case OERegion::Type::HesAff:
      os << "Hessian-Affine" << endl;
      break;
    case OERegion::Type::MSER:
      os << "MSER" << endl;
      break;
    default:
      break;
    }
    os << "Position:\t" << f.coords.transpose() << endl;
    os << "Extremum type:\t" << as_integer(f.extremum_type) << endl;
    os << "Extremum value:\t" << f.extremum_value << endl;
    os << "shape matrix:\n" << f.shape_matrix << endl;
    os << "orientation:\t" << to_degree(f.orientation) << " degrees" << endl;
    return os;
  }

  istream& operator>>(istream& in, OERegion& f)
  {
    auto feature_type = int{};
    in >> f.x() >> f.y() >> f.shape_matrix >> f.orientation >> feature_type;
    f.type = static_cast<OERegion::Type>(feature_type);
    return in;
  }

} /* namespace Sara */
} /* namespace DO */
