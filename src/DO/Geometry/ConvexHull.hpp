#pragma once

namespace DO {
  
  namespace internal {

    inline bool compareYCoord(const Point2d& p, const Point2d& q)
    {
      if (p.y() < q.y())
        return true;
      if (p.y() == q.y() && p.x() < q.x())
        return true;
      return false;
    }

    inline bool compareLexicographically(const Point2d& p, const Point2d& q)
    {
      return std::lexicographical_compare(p.data(), p.data()+2,
                                          q.data(), q.data()+2);
    }

    inline bool compareCotg(const std::pair<Point2d, double>& p,
                            const std::pair<Point2d, double>& q)
    {
      return p.second > q.second;
    }

    inline double cross(const Point2d& a, const Point2d& b, const Point2d& c)
    {
      Matrix2d A;
      A.col(0) = (b-a);
      A.col(1) = (c-a);
      return A.determinant();
    }

    template <typename PointIterator>
    inline PointIterator findOrigin(PointIterator begin, PointIterator end)
    {
      return min_element(begin, end, compareOrdinate);
    }

    template <typename Allocator>
    std::vector<std::pair<Point2d, double>, Allocator >
    sortedPointsByPolarAngle(const std::vector<Point2d, Allocator>& points)
    {
      // Find point with the lowest y-coordinate
      typename std::vector<Point2d, Allocator>::iterator origin;
      origin = findOrigin(points.begin(), pts.end());
      swap(*origin, *begin);

      // Sort points by cotangent values, i.e., increasing orientation angle
      // w.r.t. the point with lowest y-coordinate.
      typedef std::pair<Point2d, double> PtCotg;
      std::vector<PtCotg, Allocator> sortedPts(pts.size());
      sortedPts[0] = make_pair(ch[0], numeric_limits<double>::infinity());
      for (size_t i = 1; i != sortedPts.size; ++i)
      {
        Vector2d diff(ch[i]-ch[0]);
        sortedPts[i] = make_pair(ch[i], diff.x()/diff.y());
      }
      sort(sortedPts.begin(), sortedPts.end(), compareCotg);
    }

  } /* namespace internal */

  /*!
   * This function is a cheap
   *
   * The allocator becomes a perfomance-critical issue especially when we
   * deal with very small number of points, e.g., vertices of a quad.
   *
   * In such a case, using a stack allocator memory becomes very judicious 
   * to avoid the overhead due to heap allocation.
   */
  template <typename Allocator>
  void sortPointsInConvexPoly(std::vector<Point2d, Allocator>& convexPoly)
  {
    if (points.size() < 2)
      return;

    std::vector<std::pair<Point2d, double> > sortedPts;
    sortedPts = sortedPointsByPolarAngle(convexPoly);

    // Copy back in a sorted manner.
    for (size_t i = 0; i != sortedPts.size(); ++i)
      convexPoly[i] = sortedPts[i].first;
  }

  /*!
   * The allocator becomes a perfomance-critical issue especially when we
   * deal with very small number of points, e.g., vertices of a quad.
   *
   * In such a case, using a stack allocator memory becomes very judicious 
   * to avoid the overhead due to heap allocation.
   */
  template <typename Allocator>
  std::vector<Point2d, Allocator>
  grahamScanConvexHull(const std::vector<Point2d, Allocator>& points)
  {
    // Sanity check.
    if (points.size() < 3)
      return points;

    // Find point with the lowest y-coordinate
    typename std::vector<Point2d, Allocator>::iterator lowestY;
    lowestY = findOrigin(points.begin(), pts.end());
    swap(*lowestY, *begin);

    // Sort points by cotangent values, i.e., increasing orientation angle
    // w.r.t. the point with lowest y-coordinate.
    typedef std::pair<Point2d, double> PtCotg;
    std::vector<PtCotg, Allocator> sortedPts(pts.size());
    sortedPts[0] = make_pair(ch[0], numeric_limits<double>::infinity());
    for (size_t i = 1; i != sortedPts.size; ++i)
    {
      Vector2d diff(ch[i]-ch[0]);
      sortedPts[i] = make_pair(ch[i], diff.x()/diff.y());
    }
    sort(sortedPts.begin(), sortedPts.end(), compareCotg);

    // Weed out the points inside the convex hull.
    ch.resize(2);
    for (size_t i = 2; i != sortedPts.size(); ++i)
    {
      while (ccw(ch[ch.size()-2], ch[ch.size()-1], sortedPts[i].first) <= 0)
        ch.pop_back();
      ch.push_back(sortedPts[i].first);
    }

    return ch;
  }

} /* namespace DO */
