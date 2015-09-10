// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
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

    inline Cone(const Vector& alpha, const Vector& beta, Type type = Convex,
                double eps = 1e-8)
      : eps_(eps), type_(type)
    {
      basis_.col(0) = alpha.normalized();
      basis_.col(1) = beta.normalized();
      FullPivLU<Basis> luSolver(basis_);
      luSolver.setThreshold(eps_);
      if (luSolver.rank() == 1)
      {
        type_ |= Pointed;
        if (basis_.col(0).dot(basis_.col(1)) > 0)
          type_ |= PositiveCosine;
      }
    }

    inline const Basis& basis() const { return basis_; }
    inline Vector alpha() const { return basis_.col(0); }
    inline Vector beta() const { return basis_.col(1); }

    friend
    inline bool inside(const Vector& p, const Cone& K)
    { return K.contains(p); }

  protected:
    bool contains(const Vector& x) const
    {
      // Deal with the null vector.
      if (x.squaredNorm() < eps_*eps_)
        return (type_ & Blunt) != 0;

      // Otherwise decompose x w.r.t. to the basis.
      Vector2d theta;
      theta = basis_.fullPivLu().solve(x);
      double relError = (basis_*theta - x).squaredNorm() / x.squaredNorm();
      if (relError > eps_)
        return false;

      // Deal with the degenerate cases (Pointed cone).
      if (type_ & Pointed)
      {
        if (type_ & PositiveCosine && type_ & Blunt)
          return theta.minCoeff() > -eps_;
        return (type_ & Blunt) != 0;
      }

      // Generic case.
      double minCoeff = theta.minCoeff();
      if (type_ & Convex)
        return minCoeff > eps_;
      return minCoeff > -eps_;
    }

  protected:
    Basis basis_;
    double eps_;
    unsigned char type_;
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
      : Base(alpha, beta, type), vertex_(vertex) {}

    inline const Vector& vertex() const { return vertex_; }

    friend
    inline bool inside(const Vector& p, const AffineCone& K)
    { return K.contains(Vector(p-K.vertex_)); }

  private:
    Vector vertex_;
  };

  typedef Cone<2> Cone2;
  typedef Cone<3> Cone3;
  typedef AffineCone<2> AffineCone2;
  typedef AffineCone<3> AffineCone3;

  DO_SARA_EXPORT
  AffineCone2 affine_cone2(double theta0, double theta1, const Point2d& vertex);


} /* namespace Sara */
} /* namespace DO */

#endif /* DO_SARA_GEOMETRY_OBJECTS_CONE_HPP */
