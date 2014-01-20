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

#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>

using namespace std;

namespace DO {

  std::ostream& operator<<(std::ostream& os, const BBox& bbox)
  {
    os << "top-left: [" << bbox.topLeft().transpose() << "]" << endl;
    os << "bottom-right: [" << bbox.bottomRight().transpose() << "]" << endl;
    return os;
  }

  double area(const BBox& bbox)
  {
    return bbox.width()*bbox.height();
  }

  bool inside(const Point2d& p, const BBox& bbox)
  {
    return 
      p.x() >= bbox.topLeft().x() && p.x() <= bbox.bottomRight().x() &&
      p.y() >= bbox.topLeft().y() && p.y() <= bbox.bottomRight().y() ;
  }

  bool degenerate(const BBox& bbox, double eps)
  {
    return area(bbox) < eps;
  }

  bool intersect(const BBox& bbox1, const BBox& bbox2)
  {
    BBox inter(intersection(bbox1, bbox2));
    return area(inter) > std::numeric_limits<double>::epsilon();
  }

  double jaccardSimilarity(const BBox& bbox1, const BBox& bbox2)
  {
    BBox inter(intersection(bbox1, bbox2));
    double interArea = area(inter);
    double unionArea = area(bbox1) + area(bbox2) - interArea;
    return interArea/unionArea;
  }

  double jaccardDistance(const BBox& bbox1, const BBox& bbox2)
  {
    return 1. - jaccardSimilarity(bbox1, bbox2);
  }

  static
  void getCorners(Point2d& tl, Point2d& tr, Point2d& br, Point2d& bl,
                  const BBox& bbox)
  {
    tl = bbox.topLeft();
    tr = bbox.topRight();
    br = bbox.bottomRight();
    bl = bbox.bottomLeft();
  }

  static
  BBox intersectionOneWay(const BBox& bbox1, const BBox& bbox2)
  {
    Point2d tl1, tr1, br1, bl1;
    Point2d tl2, tr2, br2, bl2;
    getCorners(tl1, tr1, br1, bl1, bbox1);
    getCorners(tl2, tr2, br2, bl2, bbox2);

    // Case 1
    if (inside(tl2, bbox1))
    {
      Point2d br;
      br = br1.cwiseMin(br2);
      return BBox(tl2, br);
    }
    // Case 2
    if (inside(br2, bbox1))
    {
      Point2d tl;
      tl = tl1.cwiseMax(tl2);
      return BBox(tl, br2);
    }
    // Case 3
    if (inside(tr2, bbox1))
    {
      Point2d tl, br;
      tl << max(tl1.x(), tl2.x()), tr2.y();
      br << tr2.x(), min(br1.y(), br2.y());
      return BBox(tl, br);
    }
    // Case 4
    if (inside(bl2, bbox1))
    {
      Point2d tl, br;
      tl << bl2.x(), max(tl1.y(), tl2.y());
      br << min(br1.x(), br2.x()), bl2.y();
      return BBox(tl, br);
    }
    return BBox(Point2d::Zero(), Point2d::Zero());
  }

  BBox intersection(const BBox& bbox1, const BBox& bbox2)
  {
    BBox bbox(BBox::zero());
    bbox = intersectionOneWay(bbox1, bbox2);
    if (area(bbox) > std::numeric_limits<double>::epsilon())
      return bbox;
    bbox = intersectionOneWay(bbox2, bbox1);
    if (area(bbox) > std::numeric_limits<double>::epsilon())
      return bbox;
    return bbox;
  }

} /* namespace DO */