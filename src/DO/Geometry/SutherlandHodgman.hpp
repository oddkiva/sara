#pragma once

namespace DO {

  /*! 
   Simple implementation of Sutherland-Hodgman algorithm.
   - Polygon 'subject' must be a simple polygon, i.e., without holes.
   - Polygon 'clip' must be a convex polygon.
   */
  std::vector<Point2d> clipPolygon(const std::vector<Point2d>& subject,
                                   const std::vector<Point2d>& clip);

} /* namespace DO */