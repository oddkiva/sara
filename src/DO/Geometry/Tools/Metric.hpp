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

#ifndef DO_GEOMETRY_TOOLS_METRIC_HPP
#define DO_GEOMETRY_TOOLS_METRIC_HPP

#include <Eigen/Eigen>

namespace DO {

  template <typename T, int N>
  class SquaredRefDistance
  {
  public:
    typedef T Scalar;
    typedef Eigen::Matrix<T, N, 1> Vector, Point;
    typedef Eigen::Matrix<T, N, N> Matrix;

  public:
    inline SquaredRefDistance(const Matrix& m) : m_(m) {}
    inline const Matrix& mappedMatrix() const { return m_; }
    inline int dim() const { return N; }
    inline T operator()(const Vector& a, const Vector& b) const
    { return (b-a).dot(m_*(b-a)); }
    inline bool isQuasiIsotropic(T threshold = 0.9) const
    {
      Eigen::JacobiSVD<Matrix> svd(m_);
      return (svd.singularValues()(N-1)/svd.singularValues(0)) < threshold;
    }

  private:
    const Matrix& m_;
  };

  template <typename T, int N>
  class SquaredDistance
  {
  public:
    typedef T Scalar;
    typedef Eigen::Matrix<T, N, 1> Vector, Point;
    typedef Eigen::Matrix<T, N, N> Matrix;

  public:
    inline SquaredDistance(const Matrix& m) : m_(m) {}
    inline Matrix& mappedMatrix()
    { return m_; }
    inline const Matrix& mappedMatrix() const
    { return m_; }
    inline int dim() const
    { return N; }
    inline T operator()(const Vector& a, const Vector& b) const
    { return (b-a).dot(m_*(b-a)); }
    inline bool isQuasiIsotropic(T threshold = 0.9) const
    {
      Eigen::JacobiSVD<Matrix> svd(m_);
      return (svd.singularValues()(N-1)/svd.singularValues(0)) < threshold;
    }

  private:
    const Matrix m_;
  };

  template <typename SquaredMetric>
  class OpenBall
  {
  public:
    typedef SquaredMetric SquaredDistance;
    typedef typename SquaredDistance::Scalar T;
    typedef typename SquaredDistance::Matrix Matrix;
    typedef typename SquaredDistance::Vector Vector, Point;

    inline OpenBall(const Point& center, T radius,
                    const SquaredDistance& squaredDistance)
      : center_(center), radius_(radius), squaredDistance_(squaredDistance) {}

    inline const Point& center() const
    { return center_; }
    inline T radius() const
    { return radius_; }
    inline const SquaredDistance& squaredDistance() const
    { return squaredDistance_; }
    inline bool isInside(const Point& x) const
    { return squaredDistance(center_, x) < radius_*radius_; }

  private:
    const Point& center_;
    const T radius_;
    const SquaredDistance& squaredDistance_;
  };

} /* namespace DO */

#endif /* DO_GEOMETRY_TOOLS_METRIC_HPP */
