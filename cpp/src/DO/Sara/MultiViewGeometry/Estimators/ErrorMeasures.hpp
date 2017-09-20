#pragma once

#include <cmath>


namespace DO { namespace Sara {

  /// @brief Functor evaluating distance of a point to its epipolar line.
  class EpipolarDistance
  {
  public:
    EpipolarDistance()
    {
    }

    template <typename Fundamental, typename Point>
    auto operator()(const Fundamental& F, const Point& x, const Point& y) const
    {
      const auto right_epipolar = F.right_line(x);
      const auto d1 =
          std::abs(right_epipolar.dot(y)) / right_epipolar.head(2).norm();

      const auto left_epipolar = F.leftLine(x);
      const auto d2 =
          std::abs(left_epipolar.dot(x)) / left_epipolar.head(2).norm();

      return std::fmax(d1, d2);
    }

  };

  /// @brief Functor evaluating distance of a point to its epipolar line.
  class SampsonDistance
  {
  };

} /* namespace Sara */
} /* namespace DO */
