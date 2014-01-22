#pragma once

#include <DO/Geometry/Polygon.hpp>

namespace DO {
  
  namespace internal {

    typedef std::pair<Point2d, double> PtCotg;
    
    void sortPointsByPolarAngle(Point2d *points, PtCotg *workArray,
                                int numPoints);

  } /* namespace internal */

  std::vector<Point2d> grahamScanConvexHull(const std::vector<Point2d>& points);

} /* namespace DO */
