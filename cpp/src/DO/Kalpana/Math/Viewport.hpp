#pragma once

#include <DO/Kalpana/Math/Projection.hpp>
#include <DO/Sara/Geometry/Objects/AxisAlignedBoundingBox.hpp>


namespace DO::Kalpana {

  struct Viewport : Sara::AxisAlignedBoundingBox<int, 2>
  {
    auto aspect_ratio() const -> float
    {
      return static_cast<float>(width()) / height();
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

    auto perspective(const float fov_degrees = 60.f, const float z_min = 0.5f,
                     const float z_max = 200.f) const -> Eigen::Matrix4f
    {
      const auto ratio = aspect_ratio();
      return Kalpana::perspective(fov_degrees, ratio, z_min, z_max);
    }
  };

}  // namespace DO::Kalpana
