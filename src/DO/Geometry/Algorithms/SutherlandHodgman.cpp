#include <DO/Geometry.hpp>

namespace DO {

  bool intersection(const P2::Line& l1, const P2::Line& l2, Vector2d& u)
  {
    P2::Point P;
    P = P2::intersection(l1, l2);

    if (fabs(P(2)) < std::numeric_limits<double>::epsilon())
      return false;

    u = P.head(2) / P(2);
    return true;
  }

  static
  P2::Line lineFrom(const Vector2d& s, const Vector2d& e)
  {
    P2::Line l;
    l = P2::line(s, e);
    l /= l.maxCoeff();
    return l;
  }
  
  // Polygon 'subject' can be a polygon of any type.
  // Polygon 'clip'  must be a convex polygon.
  std::vector<Point2d> sutherlandHodgman(const std::vector<Point2d>& subject,
                                         const std::vector<Point2d>& clip)
  {
    std::vector<Point2d> in, out;
    Point2d inter;
    P2::Line clipLine;
    P2::Line inLine;

    // Initialize
    out = subject;
    const size_t M = clip.size();
    // Loop.
    for (size_t e2 = 0, e1 = M-1; e2 != M; e1 = e2++) // 'e' like edge of the clip polygon.
    {
      in = out;
      out.clear();

      const size_t N = in.size();
      for (size_t v2 = 0, v1 = N-1; v2 != N; v1 = v2++)
      {
        int ccw1 = ccw(clip[e1], clip[e2], in[v1]);
        int ccw2 = ccw(clip[e1], clip[e2], in[v2]);

        if (ccw1 ==  1 && ccw2 ==  1)
          out.push_back(in[v2]);
        else if (ccw1 ==  1 && ccw2 == -1)
        {
          clipLine = lineFrom(clip[e1], clip[e2]);
          inLine = lineFrom(in[v1], in[v2]);

          if ( intersection(clipLine, inLine, inter) )
            out.push_back(inter);
        }
        else if (ccw1 == -1 && ccw2 ==  1)
        {
          clipLine = lineFrom(clip[e1], clip[e2]);
          inLine = lineFrom(in[v1], in[v2]);
          if ( intersection(clipLine, inLine, inter) )
            out.push_back(inter);
          out.push_back(in[v2]);
        }
      }
    }

    return out;
  }
  
}