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

#ifndef DO_SARA_GEOMETRY_POLYGON_HPP
#define DO_SARA_GEOMETRY_POLYGON_HPP

#include <vector>

#include <DO/Sara/Geometry/Tools/Projective.hpp>


namespace DO { namespace Sara {

  template <std::size_t N>
  class SmallPolygon
  {
    using self_type = SmallPolygon;

  public:
    using size_type = std::size_t;
    using point_type = Point2d;
    using line_type = Vector3d;
    using iterator =  point_type *;
    using const_iterator = const point_type *;

    //! @{
    //! Constructors
    SmallPolygon() = default;

    inline SmallPolygon(const point_type *vertices)
    {
      copy_vertices(vertices);
    }

    inline SmallPolygon(const SmallPolygon& other)
    {
      copy(other);
    }
    //! @}

    //! \brief Assignment operator
    inline SmallPolygon& operator=(const SmallPolygon& other)
    {
      copy(other);
      return *this;
    }

    //! @{
    //! \brief Point accessors.
    inline point_type& operator[](size_type i)
    {
      return _v[i];
    }

    inline const point_type& operator[](size_type i) const
    {
      return _v[i];
    }
    //! @}

    //! @{
    //! \brief iterators.
    inline point_type * begin()
    {
      return _v;
    }

    inline point_type * end()
    {
      return _v+N;
    }

    inline const point_type * begin() const
    {
      return _v;
    }

    inline const point_type * end() const
    {
      return _v+N;
    }
    //! @}

    //! \brief return the number of vertices.
    inline std::size_t num_vertices() const
    {
      return N;
    }

    //! \brief Equality comparison.
    inline bool operator==(const self_type& other) const
    {
      return std::equal(_v, _v + N, other._v);
    }

    //! \brief Inequality comparison.
    inline bool operator!=(const self_type& other) const
    {
      return !this->operator==(other);
    }

  protected:
    inline void copy_vertices(const_iterator vertices)
    {
      std::copy(vertices, vertices+N, _v);
    }

    inline void copy(const self_type& other)
    {
      copy_vertices(other._v);
    }

  protected:
    point_type _v[N];
  };

  //! I/O ostream operator.
  template <std::size_t N>
  std::ostream& operator<<(std::ostream& os, const SmallPolygon<N>& poly)
  {
    typename SmallPolygon<N>::const_iterator p = poly.begin();
    for ( ; p != poly.end(); ++p)
      os << "[ " << p->transpose() << " ] ";
    return os;
  }

  //! Utility functions.
  template <std::size_t N>
  double signed_area(const SmallPolygon<N>& polygon)
  {
    // Computation derived from Green's formula
    double A = 0.;
    for (std::size_t i1 = N-1, i2 = 0; i2 < N; i1=i2++)
    {
      Matrix2d M;
      M.col(0) = polygon[i1];
      M.col(1) = polygon[i2];
      A += M.determinant();
    }
    return 0.5*A;
  }

  template <std::size_t N>
  inline double area(const SmallPolygon<N>& polygon)
  {
    return std::abs(signed_area(polygon));
  }

  //! Even-odd rule implementation.
  template <std::size_t N>
  bool inside(const Point2d& p, const SmallPolygon<N>& poly)
  {
    bool c = false;
    for (std::size_t i = 0, j = N-1; i < N; j = i++)
    {
      if ( (poly[i].y() > p.y()) != (poly[j].y() > p.y()) &&
           (p.x() <   (poly[j].x()-poly[i].x()) * (p.y()-poly[i].y())
                    / (poly[j].y()-poly[i].y()) + poly[i].x()) )
        c = !c;
    }
    return c;
  }

  template <std::size_t N>
  bool degenerate(const SmallPolygon<N>& poly,
                  double eps = std::numeric_limits<double>::epsilon())
  {
    return area(poly) < eps;
  }

  double area(const std::vector<Point2d>& polygon);

} /* namespace Sara */
} /* namespace DO */

#endif /* DO_SARA_GEOMETRY_POLYGON_HPP */
