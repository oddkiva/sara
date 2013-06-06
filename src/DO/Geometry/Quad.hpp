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

#ifndef DO_GEOMETRY_QUAD_HPP
#define DO_GEOMETRY_QUAD_HPP

namespace DO {

  struct Quad {
    Point2d a, b, c, d; // Important: must be enumerated in ccw order.
    Vector3d lineEqns[4];
    Quad() {}
    Quad(const BBox& bbox);
    Quad(const Point2d& a_, const Point2d& b_, const Point2d& c_, const Point2d& d_);

    void dilate(double step);
    void applyH(const Matrix3d& H);
    bool invertEnumerationOrder();

    BBox bbox() const;
    bool isAlmostSimilar(const Quad& quad) const;

    bool isInside(const Point2d& p) const;
    bool intersect(const Quad& quad) const;
    double overlap(const Quad& quad) const;
    
    void print() const;
    void drawOnScreen(const Color3ub& c, double scale = 1.) const;
  };

  bool readQuads(std::vector<Quad>& quads, const std::string& filePath);
  bool writeQuads(const std::vector<Quad>& quads, const std::string& filePath);

} /* namespace DO */

#endif /* DO_GEOMETRY_QUAD_HPP */