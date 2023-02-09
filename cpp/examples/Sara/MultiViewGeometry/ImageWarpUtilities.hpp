#pragma once

#include <DO/Sara/ImageProcessing/Interpolation.hpp>


namespace DO::Sara {

  template <typename CameraModel>
  auto undistortion_map(const CameraModel& camera, const Eigen::Vector2i& sizes)
      -> std::pair<Image<float>, Image<float>>
  {
    auto umap = Image<float>{sizes};
    auto vmap = Image<float>{sizes};

    const auto& w = sizes.x();
    const auto& h = sizes.y();

    for (int v = 0; v < h; ++v)
    {
      for (int u = 0; u < w; ++u)
      {
        // Backproject the pixel from the destination camera plane.
        const auto uv = Eigen::Vector2d(u, v);
        const Eigen::Vector2f uvd = camera.distort(uv).template cast<float>();

        umap(u, v) = uvd.x();
        vmap(u, v) = uvd.y();
      }
    }

    return std::make_pair(umap, vmap);
  }

  inline auto warp(const ImageView<float>& u_map,  //
                   const ImageView<float>& v_map,  //
                   const ImageView<float>& frame,
                   ImageView<float>& frame_warped)
  {
    const auto w = frame.width();
    const auto h = frame.height();
    const auto wh = w * h;

#pragma omp parallel for
    for (auto p = 0; p < wh; ++p)
    {
      // Destination pixel.
      const auto y = p / w;
      const auto x = p - w * y;

      auto xyd = Eigen::Vector2d{};
      xyd << u_map(x, y), v_map(x, y);

      const auto in_image_domain = 0 <= xyd.x() && xyd.x() < w - 1 &&  //
                                   0 <= xyd.y() && xyd.y() < h - 1;
      if (!in_image_domain)
      {
        frame_warped(x, y) = 0.f;
        continue;
      }

      const auto color = interpolate(frame, xyd);
      frame_warped(x, y) = color;
    }
  }

  inline auto warp(const ImageView<float>& u_map,  //
                   const ImageView<float>& v_map,  //
                   const ImageView<Rgb8>& frame,   //
                   ImageView<Rgb8>& frame_warped)
  {
    const auto w = frame.width();
    const auto h = frame.height();
    const auto wh = w * h;

#pragma omp parallel for
    for (auto p = 0; p < wh; ++p)
    {
      // Destination pixel.
      const auto y = p / w;
      const auto x = p - w * y;

      auto xyd = Eigen::Vector2d{};
      xyd << u_map(x, y), v_map(x, y);

      const auto in_image_domain = 0 <= xyd.x() && xyd.x() < w - 1 &&  //
                                   0 <= xyd.y() && xyd.y() < h - 1;
      if (!in_image_domain)
      {
        frame_warped(x, y) = Black8;
        continue;
      }

      auto color = interpolate(frame, xyd);
      color /= 255;

      auto color_converted = Rgb8{};
      smart_convert_color(color, color_converted);

      frame_warped(x, y) = color_converted;
    }
  }

}  // namespace DO::Sara
