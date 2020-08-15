// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <set>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace std;
using namespace DO::Sara;


struct BirdsEyeViewProjector
{
  // Intrinsic parameters.
  Eigen::Array2f image_sizes;
  Eigen::Array2f fov;

  // Output scale parameters.
  Eigen::Array2f scale = {1.f/32, 1.f/16};

  // Extrinsic data.
  Eigen::Vector3f camera = Eigen::Vector3f{0, 0, 40};
  Eigen::Vector2f vp;

  // Cached data.
  Eigen::Array2f _alphas;
  Eigen::Array2f _vp_angles;

  auto recalculate_half_fovs()
  {
    _alphas = fov / 2.f;
  }

  auto recalculate_vp_angles()
  {
    _vp_angles = (vp.array() - image_sizes / 2) / (image_sizes / 2) * _alphas;
  }

  auto half_field_of_views() const -> const Eigen::Array2f&
  {
    return _alphas;
  }

  auto vp_angles() const -> const Eigen::Array2f&
  {
    return _vp_angles;
  }

  auto gamma(const Eigen::Vector2f& xy) const
  {
    return std::atan((xy.y() - camera.y()) / (xy.x()  - camera.x()));
  }

  auto sin_gamma(const Eigen::Vector2f& xy) const
  {
    return std::sin(gamma(xy));
  }

  auto tan_gamma(const Eigen::Vector2f& xy) const
  {
    return (xy.y() - camera.y()) / (xy.x() - camera.x());
  }

  auto world_angular_coords(const Eigen::Vector2f& xy) const
      -> Eigen::Vector2f
  {
    const auto gamma_xy = gamma(xy);
    const auto theta_xy = camera.z() * sin(gamma_xy) / (xy.y() - camera.y());
    return {gamma_xy, theta_xy};
  }

  auto normalized_world_angular_coords(const Eigen::Vector2f& xy) const
      -> Eigen::Vector2f
  {
    const auto& alphas = half_field_of_views();
    return (world_angular_coords(xy).array() - vp_angles() - alphas) * 2 * alphas;
  }

  auto uv(const Eigen::Vector2f& xy) const -> Eigen::Vector2f
  {
    return normalized_world_angular_coords(xy).array() * image_sizes;
  }

  auto normalized_image_coords(const Eigen::Vector2f& uv) const
      -> Eigen::Vector2f
  {
    return uv.array() * image_sizes;
  }

  auto image_angular_coords(const Eigen::Vector2f& uv) const
      -> Eigen::Vector2f
  {
    return normalized_image_coords(uv).array() * fov + _vp_angles - _alphas;
  }

  // Retrieve the image coordinates (u, v) from the world coordinates (x, y).
  auto xy(const Eigen::Vector2f& uv) const -> Eigen::Vector2f
  {
    const auto uv_angles = image_angular_coords(uv);

    const auto cosine = std::cos(uv_angles(0));
    const auto sine = std::sin(uv_angles(0));
    const auto cotg = 1 / tan(uv_angles(1));

    const auto& z0 = camera.z();
    return z0 * cotg * Eigen::Vector2f{cosine, sine} + camera.head(2);
  }

  auto project(const ImageView<Rgb8>& src, ImageView<Rgb8>& dst) const
  {
    const auto& depth = dst.height();
    const auto& width = dst.width();
    for (auto z = 0; z < depth; ++z)
    {
      for (auto x = -width/2; x < -width/2 + width; ++x)
      {
        const Eigen::Array2f zx = Eigen::Array2f(z, x) * scale;
        const Eigen::Vector2d uv = this->uv(zx).cast<double>();
        if ((uv.array() < 0).any() ||
            (uv.array() > image_sizes.cast<double>().array() - 1).any() ||
            std::isnan(uv.x()) || std::isnan(uv.y()))
        {
          dst(x, z) = Black8;
          continue;
        }

        const Rgb8 rgb = interpolate(src, uv)
            .array()
            .round()
            .max(0).min(255)
            .cast<std::uint8_t>()
            .matrix();
        dst(depth - z, width - x - width/2) = rgb;
      }
    }
  }

//  auto project_inverse(const ImageView<Rgb8>& src, ImageView<Rgb8>& dst) const
//  {
//    for (auto v = 0; v < dst.height(); ++v)
//    {
//      for (auto u = 0; u < dst.width(); ++u)
//      {
//      }
//    }
//  }
};


auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  // const auto video_filepath =
  // "/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
  const auto video_filepath =
      //"/Users/david/Desktop/Datasets/videos/sample1.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample4.mp4"s;
  //     //"/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_projected = Image<Rgb8>{800, 800};
  auto frame_backprojected = Image<Rgb8>{frame.sizes()};

  auto projector = BirdsEyeViewProjector{};
  projector.image_sizes = frame.sizes().cast<float>();
  projector.fov = Eigen::Array2f{40, 25} * M_PI / 180.f;
  projector.vp = projector.image_sizes / 2.f;
  projector.vp.array() -= 200.f;
  projector.recalculate_vp_angles();
  projector.recalculate_half_fovs();

  // Show the local extrema.
  create_window(frame_projected.sizes());
  set_antialiasing();

  auto frames_read = 0;
  const auto skip = 0;
  while (true)
  {
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    ++frames_read;
    if (frames_read % (skip + 1) != 0)
      continue;
    SARA_CHECK(frames_read);

    projector.project(frame, frame_projected);

    display(frame_projected);
  }
}


GRAPHICS_MAIN()
{
  test_on_video();
  return 0;
}
