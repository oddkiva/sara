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

#include <DO/Core/EigenExtension.hpp>
#include <vector>

namespace DO {

  template <int N>
  class SmallPolygon
  {
  public:
    typedef typename Matrix<T, N+1, 1> LineEquation;

    inline SmallPolygon() {}
    inline SmallPolygon(const Point *vertices)
    { copy_vertices(vertices); }
    inline SmallPolygon(const SmallPolygon& other)
    { copy(other.v_); }

    inline SmallPolygon& operator=(const SmallPolygon& other);

    inline Point2d& operator[](int i) { return v_[i]; }
    inline const Point2d& operator[](int i) const { return v_[i]; }

    inline Point2d * begin() { return v_; }
    inline Point2d * end()   { return v_+N; }
    inline const Point2d * begin() const { return v_; }
    inline const Point2d * end()   const { return v_; }

  private:
    void copy_vertices(const Point2d *vertices);
    void copy(const SmallPolygon& other);

  private:
    Point2d v_[N];
    LineEquation line_eqns_[N];
  };

  bool similar(const Quad& Q1, const Quad& Q2);
  bool inside(const Point2d& p, const Quad& Q);
  bool intersect(const Quad& Q1, const Quad& Q2);
  double overlap(const Quad& Q1, const Quad& Q2);

  //! I/O ostream operator.
  std::ostream& operator<<(std::ostream& os, const Quad& Q);

  //! Graphics function.
  void drawQuad(const Quad& Q, const Color3ub& color, double scale = 1.);

  //! I/O.
  bool readQuads(std::vector<Quad>& quads, const std::string& filePath);
  bool writeQuads(const std::vector<Quad>& quads, const std::string& filePath);

  //! computation using Green-Riemann formula
  template <int N>
  double area(const SmallPolygon<N>& polygon);
  double area(const std::vector<Point2d>& polygon);

} /* namespace DO */