#pragma once

#include <DO/Kalpana/Math/Projection.hpp>


namespace DO::Kalpana {

  struct Viewport
  {
    Eigen::Vector2i top_left = Eigen::Vector2i::Zero();
    Eigen::Vector2i sizes = Eigen::Vector2i::Zero();

    auto aspect_ratio() const -> float
    {
      return static_cast<float>(sizes.x()) / sizes.y();
    }

    auto width() const -> int
    {
      return sizes.x();
    }

    auto height() const -> int
    {
      return sizes.y();
    }

    auto orthographic_projection(const float scale = 0.5f) const
        -> Eigen::Matrix4f
    {
      const auto ratio = aspect_ratio();
      return orthographic(                //
          -scale * ratio, scale * ratio,  //
          -scale, scale,                  //
          -scale, scale);
    }
  };

}  // namespace DO::Kalpana
