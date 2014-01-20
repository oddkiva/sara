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

  static P2::Line lineFrom(const Vector2d& s, const Vector2d& e)
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
    for (size_t e = 0; e != M; ++e) // 'e' like edge of the clip polygon.
    {
      in = out;
      out.clear();

      const size_t N = in.size();
      for (size_t v = 0; v != N; ++v)
      {
        int ccw_a = ccw(clip[e], clip[(e+1)%M], in[v]);
        int ccw_b = ccw(clip[e], clip[(e+1)%M], in[(v+1)%N]);

        if (ccw_a ==  1 && ccw_b ==  1)
          out.push_back(in[(v+1)%N]);
        else if (ccw_a ==  1 && ccw_b == -1)
        {
          clipLine = lineFrom(clip[e], clip[(e+1)%M]);
          inLine = lineFrom(in[v], in[(v+1)%N]);

          if ( intersection(clipLine, inLine, inter) )
            out.push_back(inter);
        }
        else if (ccw_a == -1 && ccw_b ==  1)
        {
          clipLine = lineFrom(clip[e], clip[(e+1)%M]);
          inLine = lineFrom(in[v], in[(v+1)%N]);
          if ( intersection(clipLine, inLine, inter) )
            out.push_back(inter);
          out.push_back(in[(v+1)%N]);
        }
      }
    }

    return out;
  }
}