#pragma once

#include <DO/Sara/Geometry/Objects/AxisAlignedBoundingBox.hpp>


namespace DO::Kalpana {

  struct WindowGeometry : AxisAlignedBoundingBox<int>
  {
    WindowGeometry() = default;

    WindowGeometry(const Eigen::Vector2i& sizes)
      : AxisAlignedBoundingBox<int>{Eigen::Vector2i::Zero(), sizes}
    {
    }

    auto aspect_ratio() const -> float
    {
      return static_cast<float>(width()) / height();
    }

    auto left_half() const -> AxisAlignedBoundingBox<int>
    {
      return {{0, 0}, {width() / 2, height()}};
    }

    auto right_half() const -> AxisAlignedBoundingBox<int>
    {
      return {{width() / 2, 0}, {width() / 2, height()}};
    }
  };

}  // namespace DO::Kalpana
