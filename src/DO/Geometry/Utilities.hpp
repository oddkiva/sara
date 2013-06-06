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

#ifndef DO_GEOMETRY_UTILITIES_HPP
#define DO_GEOMETRY_UTILITIES_HPP

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif

namespace DO {

  template <typename T>
  inline T toRadian(T degree)
  {
    DO_STATIC_ASSERT( !std::numeric_limits<T>::is_integer, 
      SCALAR_MUST_BE_OF_FLOATING_TYPE );
    return degree*static_cast<T>(M_PI)/static_cast<T>(180);
  }

  template <typename T>
  inline T toDegree(T radian)
  {
    DO_STATIC_ASSERT( !std::numeric_limits<T>::is_integer, 
      SCALAR_MUST_BE_OF_FLOATING_TYPE );
    return radian*static_cast<T>(180)/static_cast<T>(M_PI);
  }

  template <typename T>
  inline Matrix<T, 2, 1> unitVector2(T radian)
  {
    DO_STATIC_ASSERT( !std::numeric_limits<T>::is_integer, 
      SCALAR_MUST_BE_OF_FLOATING_TYPE );
    return Matrix<T, 2, 1>(std::cos(radian), std::sin(radian));
  }

  template <typename T>
  inline Matrix<T, 2, 2> rotation2(T radian)
  {
    DO_STATIC_ASSERT( !std::numeric_limits<T>::is_integer, 
      SCALAR_MUST_BE_OF_FLOATING_TYPE );
    return Eigen::Rotation2D<T>(radian).toRotationMatrix();
  }

  template <typename T>
  inline Matrix<T, 2, 2> isometry2(T radian, T scale)
  {
    DO_STATIC_ASSERT( !std::numeric_limits<T>::is_integer, 
      SCALAR_MUST_BE_OF_FLOATING_TYPE );
    return Eigen::Rotation2D<T>(radian).toRotationMatrix()*scale;
  }

  template <typename T>
  Matrix<T, 2, 2> linearTransform2(const Matrix<T, 2, 1>& va,
                                   const Matrix<T, 2, 1>& vb,
                                   const Matrix<T, 2, 1>& fva,
                                   const Matrix<T, 2, 1>& fvb)
  {
    Matrix<T, 4, 4> M;
    M << va.x(), va.y(), T(0)  , T(0),
         T(0)  , T(0)  , va.x(), va.y(),
         vb.x(), vb.y(), T(0)  , T(0),
         T(0)  , T(0)  , vb.x(), vb.y();

    Matrix<T, 4, 1> b;
    b << fva.x(), fva.y(), fvb.x(), fvb.y();

    Matrix<T, 4, 1> vecA(M.colPivHouseholderQr().solve(b));

    Matrix<T, 2, 2> A;
    A << vecA[0], vecA[1],
         vecA[2], vecA[3];

    return A;
  }

  template <typename T>
  Matrix<T, 3, 3> affineTransform2(const Matrix<T, 2, 1>& a,
                                   const Matrix<T, 2, 1>& b,
                                   const Matrix<T, 2, 1>& c,
                                   const Matrix<T, 2, 1>& fa,
                                   const Matrix<T, 2, 1>& fb,
                                   const Matrix<T, 2, 1>& fc)
  {
    Matrix<T, 6, 6> M;
    M << a.x(), a.y(), T(1), T(0) , T(0) , T(0),
         T(0) , T(0) , T(0), a.x(), a.y(), T(1),
         b.x(), b.y(), T(1), T(0) , T(0) , T(0),
         T(0) , T(0) , T(0), b.x(), b.y(), T(1),
         c.x(), c.y(), T(1), T(0) , T(0) , T(0),
         T(0) , T(0) , T(0), c.x(), c.y(), T(1);

    Matrix<T, 6, 1> y;
    y << fa.x(), fa.y(), fb.x(), fb.y(), fc.x(), fc.y();

    Matrix<T, 6, 1> x(M.colPivHouseholderQr().solve(y));

    Matrix<T, 3, 3> A;
    A << x[0], x[1], x[2],
         x[3], x[4], x[5],
         T(0), T(0), T(1);

    return A;
  }

  template <typename T>
  Matrix<T, 2, 2> linearPartFromAffineTransform2(const Matrix<T, 3, 3>& A)
  { return A.template block<2,2>(0,0); }

  template <typename T>
  Matrix<T, 3, 3> homography(const Matrix<T, 2, 1>& a,
                             const Matrix<T, 2, 1>& b,
                             const Matrix<T, 2, 1>& c,
                             const Matrix<T, 2, 1>& d,
                             const Matrix<T, 2, 1>& fa,
                             const Matrix<T, 2, 1>& fb,
                             const Matrix<T, 2, 1>& fc,
                             const Matrix<T, 2, 1>& fd)
  {
    Matrix<T, 8, 8> M;
    M << 
    a.x(), a.y(), T(1), T(0) , T(0) , T(0), -a.x()*fa.x(), -a.y()*fa.x(),
    T(0) , T(0) , T(0), a.x(), a.y(), T(1), -a.x()*fa.y(), -a.y()*fa.y(),
    b.x(), b.y(), T(1), T(0) , T(0) , T(0), -b.x()*fb.x(), -b.y()*fb.x(),
    T(0) , T(0) , T(0), b.x(), b.y(), T(1), -b.x()*fb.y(), -b.y()*fb.y(),
    c.x(), c.y(), T(1), T(0) , T(0) , T(0), -c.x()*fc.x(), -c.y()*fc.x(),
    T(0) , T(0) , T(0), c.x(), c.y(), T(1), -c.x()*fc.y(), -c.y()*fc.y(),
    d.x(), d.y(), T(1), T(0) , T(0) , T(0), -d.x()*fd.x(), -d.y()*fd.x(),
    T(0) , T(0) , T(0), d.x(), d.y(), T(1), -d.x()*fd.y(), -d.y()*fd.y();

    Matrix<T, 8, 1> y;
    y << fa.x(), fa.y(), fb.x(), fb.y(), fc.x(), fc.y(), fd.x(), fd.y();

    Matrix<T, 8, 1> x(M.colPivHouseholderQr().solve(y));

    Matrix<T, 3, 3> A;
    A << x[0], x[1], x[2],
         x[3], x[4], x[5],
         x[6], x[7], T(1);

    return A;
  }

  template <typename T>
  Matrix<T, 2, 2> homographyJacobianMatrix(const Matrix<T, 3, 3>& H, const Matrix<T, 2, 1>& x)
  {
    Matrix<T, 2, 2> dH;
    const T u = H(0,0)*x[0] + H(0,1)*x[1] + H(0,2);
    const T v = H(1,0)*x[0] + H(1,1)*x[1] + H(1,2);
    const T w = H(2,0)*x[0] + H(2,1)*x[1] + H(2,2);
    
    dH << (H(0,0)*w - H(2,0)*u)/(w*w), (H(1,0)*w - H(2,0)*u)/(w*w)
        (H(0,1)*w - H(2,1)*u)/(w*w), (H(1,1)*w - H(2,1)*u)/(w*w);
    
    return dH;
  }

  template <typename T>
  Matrix<T, 2, 1> apply(const Matrix<T, 3, 3> H, const Matrix<T, 2, 1>& p)
  {
    Matrix<T, 3, 1> H_p(H*(Vector3f() << p, 1.f).finished()); //.block(0,0,2,1)
    H_p /= H_p(2);
    return H_p.block(0,0,2,1);
  }

} /* namespace DO */

#endif /* DO_GEOMETRY_UTILITIES_HPP */