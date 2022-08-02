#pragma once

#include <DO/Sara/Core/PhysicalQuantities.hpp>


namespace DO::Sara {

  class Chessboard
  {
  public:
    inline Chessboard() = default;

    inline explicit Chessboard(const Eigen::Vector2i& corner_count_per_axis,
                               Length square_size_in_meters)
      : _corner_count_per_axis{corner_count_per_axis(0),
                               corner_count_per_axis(1)}
      , _square_size_in_meters{square_size_in_meters}
    {
    }

    inline auto width() const -> int
    {
      return _corner_count_per_axis.x();
    }

    inline auto height() const -> int
    {
      return _corner_count_per_axis.y();
    }

    inline auto corner_count() const -> int
    {
      return width() * height();
    }

    inline auto image_point(int x, int y) const -> const Eigen::Vector2f&
    {
      const auto index = y * width() + x;
      return _corners[index];
    }

    inline auto scene_point(int x, int y) const -> Eigen::Vector2f
    {
      return Eigen::Vector2f(_square_size_in_meters.value * x,
                             _square_size_in_meters.value * y);
    }

    inline auto corners_count_per_axis() const -> const Eigen::Vector2i&
    {
      return _corner_count_per_axis;
    }

    inline auto square_size_in_meters() const -> const Length&
    {
      return _square_size_in_meters;
    }

  private:
    Eigen::Vector2i _corner_count_per_axis;
    Length _square_size_in_meters;
    std::vector<Eigen::Vector2f> _corners;
  };

}  // namespace DO::Sara
