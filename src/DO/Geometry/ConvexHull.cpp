#include <DO/Geometry.hpp>

using namespace std;

namespace DO {
  
  namespace internal {
    
    static
    inline bool compareYCoord(const PtCotg& p, const PtCotg& q)
    {
      if (p.first.y() < q.first.y())
        return true;
      if (p.first.y() == q.first.y() && p.first.x() < q.first.x())
        return true;
      return false;
    }
    
    static
    inline bool compareCotg(const std::pair<Point2d, double>& p,
                            const std::pair<Point2d, double>& q)
    {
      return p.second > q.second;
    }
    
    static
    void sortPointsByPolarAngle(PtCotg *out, const Point2d *in, int numPoints)
    {
      // Copy.
      for (int i = 0; i < numPoints; ++i)
        out[i].first = in[i];
      // Find origin and swap with first element.
      PtCotg *origin;
      origin = min_element(out, out+numPoints, compareYCoord);
      swap(*origin, *out);
      // Compute the polar angle w.r.t. origin and sort by polar angle.
      out[0].second = numeric_limits<double>::infinity();
      for (int i = 1; i < numPoints; ++i)
      {
        Vector2d diff(out[i].first - out[0].first);
        out[i].second = diff.x()/diff.y();
      }
      // Compute the polar angle w.r.t. origin and sort by polar angle.
      sort(out, out+numPoints, compareCotg);
    }
    
    void sortPointsByPolarAngle(Point2d *inout, PtCotg *work, int numPoints)
    {
      sortPointsByPolarAngle(work, inout, numPoints);
      for (int i = 0; i < numPoints; ++i)
        inout[i] = work[i].first;
    }

  } /* namespace internal */

  std::vector<Point2d> grahamScanConvexHull(const std::vector<Point2d>& points)
  {
    using namespace internal;
    // Sanity check.
    if (points.size() < 3)
      return points;
    using namespace internal;
    // Sort by polar angle.
    vector<PtCotg> ptCotgs(points.size());
    sortPointsByPolarAngle(&ptCotgs[0], &points[0], points.size());
    // Weed out the points inside the convex hull.
    std::vector<Point2d> ch;
    ch.reserve(points.size());
    ch.push_back(ptCotgs[0].first);
    ch.push_back(ptCotgs[1].first);
    for (size_t i = 2; i != ptCotgs.size(); ++i)
    {
      while (ccw(ch[ch.size()-2], ch[ch.size()-1], ptCotgs[i].first) <= 0)
        ch.pop_back();
      ch.push_back(ptCotgs[i].first);
    }

    return ch;
  }

} /* namespace DO */
