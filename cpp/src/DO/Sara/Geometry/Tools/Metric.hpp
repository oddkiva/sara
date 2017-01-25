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

#ifndef DO_SARA_GEOMETRY_TOOLS_METRIC_HPP
#define DO_SARA_GEOMETRY_TOOLS_METRIC_HPP

#include <Eigen/Eigen>

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  template <typename T, int N>
  class SquaredRefDistance
  {
  public:
    enum { Dim = N };

    using Scalar = T;
    using Vector = Eigen::Matrix<T, N, 1>;
    using Matrix = Eigen::Matrix<T, N, N>;

  public:
    inline SquaredRefDistance(const Matrix& m)
      : _covariance_matrix(m)
    {
    }

    inline const Matrix& covariance_matrix() const
    {
      return _covariance_matrix;
    }

    inline T operator()(const Vector& a, const Vector& b) const
    {
      return (b-a).dot(_covariance_matrix*(b-a));
    }

    inline bool is_quasi_isotropic(T threshold = 0.9) const
    {
      Eigen::JacobiSVD<Matrix> svd(_covariance_matrix);
      const Vector S = svd.singularValues();
      return S(N - 1) / S(0) > threshold;
    }

  private:
    const Matrix& _covariance_matrix;
  };

  template <typename T, int N>
  class SquaredDistance
  {
  public:
    enum { Dim = N };

    using Scalar = T;
    using Vector = Eigen::Matrix<T, N, 1>;
    using Matrix = Eigen::Matrix<T, N, N>;

  public:
    inline SquaredDistance(const Matrix& m)
      : _m(m)
    {
    }

    inline const Matrix& covariance_matrix() const
    {
      return _m;
    }

    inline T operator()(const Vector& a, const Vector& b) const
    {
      return (b-a).dot(_m*(b-a));
    }

    inline bool is_quasi_isotropic(T threshold = 0.9) const
    {
      Eigen::JacobiSVD<Matrix> svd(_m);
      const Vector S = svd.singularValues();
      return S(N - 1) / S(0) > threshold;
    }

  private:
    const Matrix _m;
  };

  template <typename SquaredMetric>
  class OpenBall
  {
  public:
    using SquaredDistance = SquaredMetric;
    using T = typename SquaredDistance::Scalar;
    using Matrix = typename SquaredDistance::Matrix;
    using Vector = typename SquaredDistance::Vector;
    using Point = Vector;

    inline OpenBall(const Point& center, T radius,
                    const SquaredDistance& squared_distance)
      : _center(center)
      , _radius(radius)
      , _squared_distance(squared_distance)
    {
    }

    inline const Point& center() const
    {
      return _center;
    }

    inline T radius() const
    {
      return _radius;
    }

    inline const SquaredDistance& squared_distance() const
    {
      return _squared_distance;
    }

    inline bool contains(const Point& x) const
    {
      return _squared_distance(x, _center) < _radius*_radius;
    }

  private:
    const Point _center;
    const T _radius;
    const SquaredDistance& _squared_distance;
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_TOOLS_METRIC_HPP */
