#pragma once

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/ImageProcessing/Warp.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/BrownConradyCamera.hpp>


namespace DO::Sara {


  class ImageDistortionCorrector
  {
  public:
    ImageDistortionCorrector(
        const ImageView<Rgb8>& image_rgb8,     //
        const ImageView<float>& image_gray32f,  //
        const v2::BrownConradyDistortionModel<double>& camera)
      : _rgb8{image_rgb8}
      , _gray32f{image_gray32f}
    {
      std::tie(_umap, _vmap) = generate_undistortion_map(camera,  //
                                                         image_rgb8.sizes());
      for (auto i = 0; i < 2; ++i)
      {
        _rgb8_undistorted[i].resize(image_rgb8.sizes());
        _gray32f_undistorted[i].resize(image_rgb8.sizes());
      }
    }

    auto undistort() -> void
    {
      _gray32f_undistorted.front().swap(_gray32f_undistorted.back());
      warp(_umap, _vmap, _gray32f, _gray32f_undistorted.back());

      _rgb8_undistorted.front().swap(_rgb8_undistorted.back());
      warp(_umap, _vmap, _rgb8, _rgb8_undistorted.back());
    }

    auto frame_gray32f(int i = 1) const -> const ImageView<float>&
    {
      return _gray32f_undistorted[i];
    }

    auto frame_rgb8(int i = 1) const -> const ImageView<Rgb8>&
    {
      return _rgb8_undistorted[i];
    }

    template <typename CameraModel>
    static auto generate_undistortion_map(const CameraModel& camera,
                                          const Eigen::Vector2i& sizes)
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

  private:
    const ImageView<Rgb8>& _rgb8;
    const ImageView<float>& _gray32f;

    Image<float> _umap;
    Image<float> _vmap;
    std::array<Image<Rgb8>, 2> _rgb8_undistorted;
    std::array<Image<float>, 2> _gray32f_undistorted;
  };

}  // namespace DO::Sara
