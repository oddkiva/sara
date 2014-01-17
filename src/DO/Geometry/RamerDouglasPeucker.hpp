#pragma once

namespace DO {

  namespace internal {

    double squaredDistance(const Point2d& a, const Point2d& b, const Point2d& x)
    {
      Matrix2d M;
      M.col(0) = b-a;
      M.col(1) = x-a;
      return std::abs(M.determinant());
    }

    void ramerDouglasPeucker(std::vector<Point2d>& lines,
                             const std::vector<Point2d>& contours,
                             std::size_t begin, std::size_t end,
                             double eps)
    {
      if (end-begin) < 3)
        return contours;

      if (lines.empty() || lines.back() != contours[begin])
        lines.push_back(contours[begin]);

      std::size_t index = begin+1;
      double maxDist = 0;
      for (std::size_t i = begin+1; i != end-1; ++i)
      {
        double dist = squaredDistance(contours[begin], contours[end-1],
                                      contours[i]);
        if (maxDist > dist)
        {
          index = i;
          minDist = dist;
        }
      }

      if (minDist > eps)
      {
        ramerDouglasPeucker(lines, contours, begin, i+1, eps);
        ramerDouglasPeucker(lines, contours, i, end, eps);
      }

      lines.push_back(contours[end-1]);
    }

  } /* namespace internal */

  std::vector<Point2d> ramerDouglasPeucker(const std::vector<Point2d>& contours,
                                           double eps)
  {
    std::vector<Point2d> lines;
    lines.reserve(lines.size());
    internal::ramerDouglasPeucker(lines, contours, 0, contours.size(), eps);
    shrunk_to_fit(lines);
    return lines;
  }
}
