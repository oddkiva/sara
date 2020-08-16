// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Geometry.hpp>


namespace DO { namespace Sara {

  bool intersection(const Projective::Line2<double>& l1,
                    const Projective::Line2<double>& l2,
                    Eigen::Vector2d& u)
  {
    const auto P = Projective::intersection(l1, l2);

    if (fabs(P(2)) < std::numeric_limits<double>::epsilon())
      return false;

    u = P.hnormalized();
    return true;
  }

  inline auto line(const Point2d& s, const Point2d& e)
  {
    auto l = Projective::line(s.homogeneous().eval(), e.homogeneous().eval());
    l /= l.maxCoeff();
    return l;
  }

  // Polygon 'subject' can be a polygon of any type.
  // Polygon 'clip'  must be a convex polygon.
  std::vector<Point2d> sutherland_hodgman(const std::vector<Point2d>& subject,
                                          const std::vector<Point2d>& clip)
  {
    auto in = std::vector<Point2d>{};
    auto out = std::vector<Point2d>{};
    auto inter = Point2d{};
    auto clip_line = Projective::Line2<double>{};
    auto in_line = Projective::Line2<double>{};

    // Initialize
    out = subject;
    const auto M = clip.size();
    // Loop.
    for (auto e2 = std::size_t{}, e1 = M - 1; e2 != M;
         e1 = e2++)  // 'e' like edge of the clip polygon.
    {
      in = out;
      out.clear();

      const auto N = in.size();
      for (auto v2 = std::size_t{}, v1 = N - 1; v2 != N; v1 = v2++)
      {
        const auto ccw1 = ccw(clip[e1], clip[e2], in[v1]);
        const auto ccw2 = ccw(clip[e1], clip[e2], in[v2]);

        if (ccw1 == 1 && ccw2 == 1)
          out.push_back(in[v2]);
        else if (ccw1 == 1 && ccw2 == -1)
        {
          clip_line = line(clip[e1], clip[e2]);
          in_line = line(in[v1], in[v2]);

          if (intersection(clip_line, in_line, inter))
            out.push_back(inter);
        }
        else if (ccw1 == -1 && ccw2 == 1)
        {
          clip_line = line(clip[e1], clip[e2]);
          in_line = line(in[v1], in[v2]);
          if (intersection(clip_line, in_line, inter))
            out.push_back(inter);
          out.push_back(in[v2]);
        }
      }
    }

    return out;
  }

}}  // namespace DO::Sara
