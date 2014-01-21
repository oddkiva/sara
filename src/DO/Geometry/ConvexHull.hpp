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

    template <typename PointIterator>
    inline PointIterator findOrigin(PointIterator begin, PointIterator end)
    {
      return min_element(begin, end, compareYCoord);
    }

    template <typename PtCotgIterator, typename PointIterator>
    void sortByPolarAngle(PtCotgIterator outBegin, PtCotgIterator outEnd,
                          PointIterator inBegin, PointIterator inEnd)
    {
      // Sanity check
      if (outEnd-outBegin != inEnd-inBegin)
      {
        const char *msg = "output array size and input array size differ";
        throw std::logic_error(msg);
      }
      // Avoid degenerate cases.
      if (outEnd-outBegin < 3)
        return;
      // Find the origin, i.e., the point with the lowest y-coordinate.
      PointIterator origin = findOrigin(inBegin, inEnd);
      swap(*origin, *inBegin);
      // Sort points by cotangent values, i.e., increasing orientation angle
      // w.r.t. the point with lowest y-coordinate.
      typedef std::pair<Point2d, double> PtCotg;
      *outBegin = make_pair(*inBegin, numeric_limits<double>::infinity());
      PointIterator i = inBegin;
      PtCotgIterator o = outBegin;
      for (++i, ++o; o != outEnd; ++i, ++o)
      {
        Vector2d diff(*i - *inBegin);
        *o = make_pair(*i, diff.x()/diff.y());
      }
      sort(outBegin, outEnd, compareCotg);
    }

  } /* namespace internal */

  void sortByPolarAngle(std::vector<Point2d>& points)
  {
    std::vector<std::pair<Point2d, double> > ptCotgs(points.size());
    sortByPolarAngle(ptCotgs.begin(), ptCotgs.end(),
                     points.begin(), points.end());
    for (size_t i = 0; i != ptCotgs.size(); ++i)
      points[i] = sortedPts[i].first;
  }

  template <int N>
  void sortByPolarAngle(Point2d *points[])
  {
    std::pair<Point2d, double> > ptCotgs[N];
    sortedPointsByPolarAngle(ptCotgs, ptCotgs+N, points, points+N);
    for (size_t i = 0; i != N; ++i)
      points[i] = sortedPts[i].first;
  }

  std::vector<Point2d> grahamScanConvexHull(const std::vector<Point2d>& points)
  {
    // Sanity check.
    if (points.size() < 3)
      return points;
    // Sort by polar angle.
    std::vector<std::pair<Point2d, double> > sortedPts(points.size());
    sortByPolarAngle(sortedPts.begin(), sortedPts.end(),
                     points.begin(), points.end());
    // Weed out the points inside the convex hull.
    std::vector<Point2d> ch;
    ch.reserve(points.size());
    ch.push_back(sortedPts[0].first);
    ch.push_back(sortedPts[1].first);
    for (size_t i = 2; i != sortedPts.size(); ++i)
    {
      while (cross(ch[ch.size()-2], ch[ch.size()-1], sortedPts[i].first) <= 0)
        ch.pop_back();
      ch.push_back(sortedPts[i].first);
    }

    return ch;
  }

} /* namespace DO */
