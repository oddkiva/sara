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

#ifndef DO_REGIONGROWING_BBOX_HPP
#define DO_REGIONGROWING_BBOX_HPP

namespace DO {

  class BBox
  {
    Point2d tl_, br_;
  public:
    BBox() {}
    BBox(const Point2d& topLeft, const Point2d& bottomRight)
      : tl_(topLeft), br_(bottomRight)
    {
      if ( tl_.x() > br_.x() ||
           tl_.y() > br_.y() )
      {
        throw std::runtime_error(
          "Error: Top-left corners and bottom-right corners are wrong!");
      }
    }

    void drawOnScreen(const Color3ub& c, double scale = 1.,
                      int thickness = 1) const;
    void print() const;
    
    Point2d& topLeft()     { return tl_; }
    Point2d& bottomRight() { return br_; }
    double& x1() { return  tl_.x(); }
    double& y1() { return  tl_.y(); }
    double& x2() { return  br_.x(); }
    double& y2() { return  br_.y(); }

    const Point2d& topLeft()     const { return tl_; }
    const Point2d& bottomRight() const { return br_; }
    Point2d        topRight()    const { return tl_+Point2d(width(), 0); }
    Point2d        bottomLeft()  const { return br_-Point2d(width(), 0); }

    double x1() const { return  tl_.x(); }
    double y1() const { return  tl_.y(); }
    double x2() const { return  br_.x(); }
    double y2() const { return  br_.y(); }

    double width() const  { return std::abs(br_.x() - tl_.x()); }
    double height() const { return std::abs(br_.y() - tl_.y()); }
    double area() const   { return width()*height(); }

    static BBox infinite() {
      BBox b;
      b.topLeft().fill(-std::numeric_limits<double>::infinity());
      b.bottomRight().fill(std::numeric_limits<double>::infinity());
      return b;
    }

    static BBox zero() {
        BBox b(Point2d::Zero(), Point2d::Zero());
        return b;
    }
  };

  BBox intersection(const BBox& bbox1, const BBox& bbox2);
  bool intersect(const BBox& bbox1, const BBox& bbox2);

  bool isSimilar(const BBox& bbox1, const BBox& bbox2,
                 double jaccardDistance);
  bool isDegenerate(const BBox& bbox, double areaThres);
  bool isInside(const Point2d& p, const BBox& bbox);

} /* namespace DO */

#endif /* DO_REGIONGROWING_BBOX_HPP */