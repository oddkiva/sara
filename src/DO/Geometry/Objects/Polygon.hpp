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
    inline SmallPolygon(const Point *vertices) { copy_vertices(vertices); }
    inline SmallPolygon(const SmallPolygon& other) { copy(other); }
    //! Assignment operator
    inline SmallPolygon& operator=(const SmallPolygon& other)
    { copy(other); return *this; }
    //! Point accessors.
    inline Point& operator[](int i) { return v_[i]; }
    inline const Point& operator[](int i) const { return v_[i]; }
    //! iterators
    inline Point * begin() { return v_; }
    inline Point * end()   { return v_+N; }
    inline const Point * begin() const { return v_; }
    inline const Point * end()   const { return v_+N; }

    inline int num_vertices() const { return N; }

  protected:
    inline void copy_vertices(const Point2d *vertices)
    { std::copy(vertices, vertices+N, v_); }
    inline void copy(const SmallPolygon& other)
    { copy_vertices(other.v_); }

  protected:
    Point v_[N];
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
  double signedArea(const SmallPolygon<N>& polygon)
  {
    //! Computation derived from Green's formula
    double A = 0.;
    for (int i1 = N-1, i2 = 0; i2 < N; i1=i2++)
    {
      Matrix2d M;
      M.col(0) = polygon[i1];
      M.col(1) = polygon[i2];
      A += M.determinant();
    }
    return 0.5*A;
  }
  
  template <int N>
  inline double area(const SmallPolygon<N>& polygon)
  { return std::abs(signedArea(polygon)); }

  //! Even-odd rule implementation.
  template <int N>
  bool inside(const Point2d& p, const SmallPolygon<N>& poly)
  {
    bool c = false;
    for (int i = 0, j = N-1; i < N; j = i++)
    {
      if ( (poly[i].y() > p.y()) != (poly[j].y() > p.y()) &&
           (p.x() <   (poly[j].x()-poly[i].x()) * (p.y()-poly[i].y())
                    / (poly[j].y()-poly[i].y()) + poly[i].x()) )
        c = !c;
    }
    return c;
  }

  template <int N>
  bool degenerate(const SmallPolygon<N>& poly,
                  double eps = std::numeric_limits<double>::epsilon())
  {
    return area(poly) < eps;
  }
  
  double area(const std::vector<Point2d>& polygon);

} /* namespace DO */

#endif /* DO_GEOMETRY_POLYGON_HPP */