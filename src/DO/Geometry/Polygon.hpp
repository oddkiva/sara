// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_GEOMETRY_POLYGON_HPP
#define DO_GEOMETRY_POLYGON_HPP

#include <DO/Geometry/Tools/Projective.hpp>
#include <vector>
#include <iostream>

namespace DO {

  template <int N>
  class SmallPolygon
  {
  public:
    typedef Point2d Point;
    typedef Vector3d Line;
    typedef Point * iterator;
    typedef const Point * const_iterator;
    //! Constructors
    inline SmallPolygon() {}
    inline SmallPolygon(const Point *vertices)
    { 
      copy_vertices(vertices);
      compute_lines();
    }
    inline SmallPolygon(const SmallPolygon& other)
    { copy(other.v_); }

    //! Assignment operator
    inline SmallPolygon& operator=(const SmallPolygon& other);
    //! Point accessors.
    inline const Point& operator[](int i) const { return v_[i]; }
    //! iterators
    inline Point * begin() { return v_; }
    inline Point * end()   { return v_+N; }
    inline const Point * begin() const { return v_; }
    inline const Point * end()   const { return v_+N; }

    inline int num_vertices() const { return N; }
    
    friend bool inside(const Point2d& p, const SmallPolygon<N>& poly);

  protected:
    inline void copy_vertices(const Point2d *vertices)
    { std::copy(vertices, vertices+N, v_); }
    inline void copy_lines(const Line *lines)
    { std::copy(lines, lines+N, lines_); }
    inline void copy(const SmallPolygon& other)
    {
      copy_vertices(other.v_);
      copy_lines(other.lines_);
    }
    inline void compute_lines() 
    {
      for (int i = 0; i < N; ++i)
      {
        lines_[i] = P2::line(v_[i], v_[(i+1)%N]);
        lines_[i] /= lines_[i](2);
      }
    }

  protected:
    Point v_[N];
    Line lines_[N];
  };

  //! I/O ostream operator.
  template <int N>
  std::ostream& operator<<(std::ostream& os, const SmallPolygon<N>& poly)
  {
    typename SmallPolygon<N>::const_iterator p = poly.begin();
    for ( ; p != poly.end(); ++p)
      os << "[ " << p->transpose() << " ] ";
    return os;
  }

  //! Utility functions.
  template <int N>
  double area(const SmallPolygon<N>& polygon)
  {
    //! Computation derived from Green's formula
    double A = 0.;
    for (int i = 0; i < N; ++i)
    {
      Matrix2d M;
      M.col(0) = polygon[i];
      M.col(1) = polygon[(i+1)%N];
      A += M.determinant();
    }
    return 0.5*A;
  }

  template <int N>
  bool inside(const Point2d& p, const SmallPolygon<N>& poly)
  {
    Vector3d P(P2::homogeneous(p));
    for (int i = 0; i < N; ++i)
      if (poly.line_eqns_[i].dot(P) < 0.)
        return false;
    return true;
  }

  template <int N>
  bool degenerate(const SmallPolygon<N>& bbox,
                  double eps = std::numeric_limits<double>::epsilon())
  {
    return area(bbox) < eps;
  }

  //void Quad::applyH(const Matrix3d& H)
  //{
  //  Vector3d ha = H*(Vector3d() << a, 1.).finished(); ha/=ha(2);
  //  Vector3d hb = H*(Vector3d() << b, 1.).finished(); hb/=hb(2);
  //  Vector3d hc = H*(Vector3d() << c, 1.).finished(); hc/=hc(2);
  //  Vector3d hd = H*(Vector3d() << d, 1.).finished(); hd/=hd(2);

  //  a = ha.head<2>();
  //  b = hb.head<2>();
  //  c = hc.head<2>();
  //  d = hd.head<2>();

  //  lineEqns[0] = computeLineEqn(a, b);
  //  lineEqns[1] = computeLineEqn(b, c);
  //  lineEqns[2] = computeLineEqn(c, d);
  //  lineEqns[3] = computeLineEqn(d, a);
  //}

  //void Quad::dilate(double step)
  //{
  //  Point2d center;
  //  center = (a+b+c+d)/4.;
  //  a += (a-center)/(a-center).norm() * step;
  //  b += (b-center)/(b-center).norm() * step;
  //  c += (c-center)/(c-center).norm() * step;
  //  d += (d-center)/(d-center).norm() * step;
  //}

  //bool Quad::invertEnumerationOrder()
  //{
  //  Point2d center;
  //  center = (a+b+c+d)/4.;
  //  if (!isInside(center))
  //  {
  //    std::swap(b,c);
  //    std::swap(d,a);
  //    lineEqns[0] = computeLineEqn(a, b);
  //    lineEqns[1] = computeLineEqn(b, c);
  //    lineEqns[2] = computeLineEqn(c, d);
  //    lineEqns[3] = computeLineEqn(d, a);
  //    return true;
  //  }
  //  if (!isInside(center))
  //    cerr << "Error: please check that the quad." << endl;
  //  return false;
  //}

  //BBox Quad::bbox() const
  //{
  //  Point2d pts[4] = { a, b, c, d };
  //  Point2d tl(a), br(a);
  //  for (int i = 0; i < 4; ++i)
  //  {
  //    if (tl.x() > pts[i].x())
  //      tl.x() = pts[i].x();
  //    if (tl.y() > pts[i].y())
  //      tl.y() = pts[i].y();

  //    if (br.x() < pts[i].x())
  //      br.x() = pts[i].x();
  //    if (br.y() < pts[i].y())
  //      br.y() = pts[i].y();
  //  }
  //  
  //  return BBox(tl, br);
  //}

  //bool Quad::isAlmostSimilar(const Quad& quad) const
  //{
  //  double thres = 1e-3;
  //  double thres2 = thres*thres;
  //  double da, db, dc, dd;
  //  da = (a-quad.a).squaredNorm();
  //  db = (b-quad.b).squaredNorm();
  //  dc = (c-quad.c).squaredNorm();
  //  dd = (d-quad.d).squaredNorm();
  //  return (da < thres2 && db < thres2 && dc < thres2 && dd < thres2);
  //}

  //bool Quad::isInside(const Point2d& p) const
  //{
  //  double signs[4];
  //  for (int i = 0; i < 4; ++i)
  //    signs[i] = lineEqns[i].dot( (Vector3d() << p, 1.).finished() );

  //  for (int i = 0; i < 4; ++i)
  //    if(signs[i] > 0)
  //      return false;
  //  return true;
  //}

  //bool Quad::intersect(const Quad& quad) const
  //{
  //  return true;
  //}

  //double Quad::overlap(const Quad& quad) const
  //{
  //  if (isAlmostSimilar(quad))
  //    return 1.;

  //  return 0.;  
  //}

  //std::ostream& operator<<(std::ostream& os, const Quad& Q)
  //{
  //  os << "a = " << Q[0].transpose() << endl;
  //  os << "b = " << Q[1].transpose() << endl;
  //  os << "c = " << Q[2].transpose() << endl;
  //  os << "d = " << Q[3].transpose() << endl;
  //  return os;
  //}

} /* namespace DO */

#endif /* DO_GEOMETRY_POLYGON_HPP */