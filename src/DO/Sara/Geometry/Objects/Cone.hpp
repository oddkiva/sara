// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_GEOMETRY_OBJECTS_CONE_HPP
#define DO_SARA_GEOMETRY_OBJECTS_CONE_HPP

#include <algorithm>

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  template <int N>
  class Cone
  {
  public:
    enum { Dimension = N };
    typedef Matrix<double, N, 2> Basis;
    typedef Matrix<double, N, 1> Vector;

    enum Type {
      Convex = 0x1, Blunt = 0x2, Pointed = 0x4,
      PositiveCosine = 0x8
    };

    //! @brief Constructor.
    inline Cone(const Vector& alpha, const Vector& beta, Type type = Convex,
                double eps = 1e-8)
      : _eps{ eps }
      , _type{ static_cast<unsigned char>(type) }
    {
      _basis.col(0) = alpha.normalized();
      _basis.col(1) = beta.normalized();
      FullPivLU<Basis> lu_solver(_basis);
      lu_solver.setThreshold(_eps);
      if (lu_solver.rank() == 1)
      {
        _type |= static_cast<unsigned char>(Pointed);
        if (_basis.col(0).dot(_basis.col(1)) > 0)
          _type |= static_cast<unsigned char>(PositiveCosine);
      }
    }

    //! @{
    //! @brief Data member accessor.
    inline const Basis& basis() const { return _basis; }

    inline Vector alpha() const { return _basis.col(0); }

    inline Vector beta() const { return _basis.col(1); }
    //! @}

    bool contains(const Vector& x) const
    {
      // Deal with the null vector.
      if (x.squaredNorm() < _eps*_eps)
        return (_type & Blunt) != 0;

      // Otherwise decompose x w.r.t. to the basis.
      Vector2d theta;
      theta = _basis.fullPivLu().solve(x);
      double rel_error = (_basis*theta - x).squaredNorm() / x.squaredNorm();
      if (rel_error > _eps)
        return false;

      // Deal with the degenerate cases (Pointed cone).
      if (_type & Pointed)
      {
        if (_type & static_cast<unsigned char>(PositiveCosine) &&
            _type & static_cast<unsigned char>(Blunt))
          return theta.minCoeff() > -_eps;
        return (_type & static_cast<unsigned char>(Blunt)) != 0;
      }

      // Generic case.
      double min_coeff = theta.minCoeff();
      if (_type & Convex)
        return min_coeff > _eps;
      return min_coeff > -_eps;
    }

  protected:
    Basis _basis;
    double _eps;
    unsigned char _type;
  };

  template <int N>
  class AffineCone : public Cone<N>
  {
    typedef Cone<N> Base;

  public:
    enum { Dimension = Base::Dimension };
    typedef typename Base::Type Type;
    typedef typename Base::Vector Vector;

    inline AffineCone(const Vector& alpha, const Vector& beta,
                      const Vector& vertex, Type type = Base::Convex)
      : Base{ alpha, beta, type }
      , _vertex{ vertex }
    {
    }

    inline const Vector& vertex() const
    {
      return _vertex;
    }

    inline bool contains(const Vector& p) const
    {
      return Base::contains(Vector(p - _vertex));
    }

  private:
    Vector _vertex;
  };

  using Cone2 = Cone<2>;
  using Cone3 = Cone<3>;
  using AffineCone2 = AffineCone<2>;
  using AffineCone3 = AffineCone<3>;

  DO_SARA_EXPORT
  AffineCone2 affine_cone2(double theta0, double theta1, const Point2d& vertex);


} /* namespace Sara */
} /* namespace DO */

#endif /* DO_SARA_GEOMETRY_OBJECTS_CONE_HPP */
