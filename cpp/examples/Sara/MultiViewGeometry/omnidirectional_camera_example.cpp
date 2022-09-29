// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/OmnidirectionalCamera.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <omp.h>


namespace sara = DO::Sara;


auto make_omnidirectional_camera()
{
  auto camera_parameters = sara::OmnidirectionalCamera<float>{};

  const auto w = 1920;
  const auto h = 1080;

  // Focal lengths in each dimension.
  const auto fx = 1063.30738864f;
  const auto fy = 1064.20554291f;
  // Shear component.
  const auto s = -1.00853432f;
  // Principal point.
  const auto u0 = 969.55702157f;
  const auto v0 = 541.26230733f;

  camera_parameters.image_sizes << w, h;
  // clang-format off
  camera_parameters.set_calibration_matrix((Eigen::Matrix3f{} <<
      fx,  s, u0,
       0, fy, v0,
       0,  0,  1).finished());
  camera_parameters.radial_distortion_coefficients <<
      0.50776095f,
      -0.16478652f,
      0.0f;
  camera_parameters.tangential_distortion_coefficients <<
      0.00023093f,
      0.00078712f;
  // clang-format on
  camera_parameters.xi = 1.50651524f;

  return camera_parameters;
}

auto make_pinhole_camera(const sara::OmnidirectionalCamera<float>& omni_camera)
{
  auto K = omni_camera.K;

  constexpr auto downscale_factor = 3.5f;
  K(0, 0) /= downscale_factor;
  K(1, 1) /= downscale_factor;
  K(1, 2) -= 450;

  const auto t = static_cast<float>(M_PI / 12);
  const auto c = std::cos(t);
  const auto s = std::sin(t);
  // clang-format off
  const auto R = (Eigen::Matrix3f{} <<
                  1,  0,  0,
                  0,  c, -s,
                  0,  s,  c).finished();
  // clang-format on

  const auto pinhole_camera = sara::PinholeCamera<float>{
      omni_camera.image_sizes,  //
      K,                        //
      K.inverse()               //
  };

  return std::make_pair(pinhole_camera, R);
}


auto undistort_image(const sara::ImageView<sara::Rgb8>& frame,
                     sara::ImageView<sara::Rgb8>& frame_undistorted,
                     const sara::OmnidirectionalCamera<float>& camera,
                     float x_min, float y_min, float scale)
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

    const auto xu = x_min + x * scale;
    const auto yu = y_min + y * scale;
    const auto xyu = Eigen::Vector2f(xu, yu);

    const Eigen::Vector2d xyd = camera.distort(xyu).cast<double>();
    const auto in_image_domain = 0 < xyd.x() && xyd.x() < w - 1 &&  //
                                 0 < xyd.y() && xyd.y() < h - 1;
    if (!in_image_domain)
    {
      frame_undistorted(x, y) = sara::Black8;
      continue;
    }

    auto color = sara::interpolate(frame, xyd);
    color /= 255;

    auto color_converted = sara::Rgb8{};
    sara::smart_convert_color(color, color_converted);

    frame_undistorted(x, y) = color_converted;
  }
}


auto flag_behind_camera(const sara::OmnidirectionalCamera<float>& camera)
    -> sara::Image<std::uint8_t>
{
  const auto w = static_cast<int>(camera.image_sizes.x());
  const auto h = static_cast<int>(camera.image_sizes.y());

  auto is_behind_camera = sara::Image<std::uint8_t>{w, h};

  for (auto y = 0; y < h; ++y)
  {
    for (auto x = 0; x < w; ++x)
    {
      const auto ray = camera.backproject(Eigen::Vector2f(x, y));
      is_behind_camera(x, y) = ray.z() < 0 ? 255 : 0;
    }
  }

  return is_behind_camera;
}

auto flag_behind_camera(const sara::ImageView<sara::Rgb8>& frame,
                        const sara::ImageView<std::uint8_t>& is_behind_camera,
                        sara::ImageView<sara::Rgb8>& is_behind_camera_map)
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

    auto c = sara::Rgb32f{};
    smart_convert_color(frame(x, y), c);
    if (is_behind_camera(x, y))
      c = 0.5f * c + 0.5f * sara::Rgb32f(0, 0, 0);

    auto c8 = sara::Rgb8{};
    smart_convert_color(c, c8);

    is_behind_camera_map(x, y) = c8;
  }
}


auto stereographic_projection(
    const sara::OmnidirectionalCamera<float>& camera_src,
    const sara::PinholeCamera<float>& camera_dst,
    const Eigen::Matrix3f& camera_rotation_dst)
{
  const auto size = camera_dst.image_sizes.cast<int>();
  auto coords_map = sara::Image<Eigen::Vector2f>{size};

  const Eigen::Matrix3f R_inverse = camera_rotation_dst.transpose();

  const auto& w = size.x();
  const auto& h = size.y();

  for (int v = 0; v < h; ++v)
  {
    for (int u = 0; u < w; ++u)
    {
      // Backproject the pixel from the destination camera plane.
      const auto uv = Eigen::Vector2f(u, v);
      const Eigen::Vector2f xy_dst = camera_dst.backproject(uv).head(2);

      // Retrieve the unit ray.
      auto ray = Eigen::Vector3f{};

      // Solve a polynomial in y.
      const auto xy_dst_squared_norm = xy_dst.squaredNorm();
      const auto a = xy_dst_squared_norm + 4;
      const auto b = -2 * xy_dst_squared_norm;
      const auto c = xy_dst_squared_norm - 4;
      const auto delta = b * b - 4 * a * c;

      const auto y = (-b - std::sqrt(delta)) / (2 * a);
      const auto x = xy_dst.x() * (1 - y) / 2;
      const auto w = xy_dst.y() * (1 - y) / 2;

      // The ray has already a norm equal to 1 by construction.
      ray << x, y, w;

      // Re-express the ray w.r.t. to the source camera frame.
      ray = R_inverse * ray;
      const auto pixel_coords = camera_src.project(ray);

      // Store it in the lookup table.
      coords_map(u, v) = pixel_coords;
    }
  }

  return coords_map;
}

auto warp(const sara::ImageView<Eigen::Vector2f>& coords_map,  //
          const sara::ImageView<sara::Rgb8>& frame,
          sara::ImageView<sara::Rgb8>& frame_warped)
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

    const Eigen::Vector2d xyd = coords_map(x, y).cast<double>();

    const auto in_image_domain = 0 <= xyd.x() && xyd.x() < w - 1 &&  //
                                 0 <= xyd.y() && xyd.y() < h - 1;
    if (!in_image_domain)
    {
      frame_warped(x, y) = sara::Black8;
      continue;
    }

    auto color = sara::interpolate(frame, xyd);
    color /= 255;

    auto color_converted = sara::Rgb8{};
    sara::smart_convert_color(color, color_converted);

    frame_warped(x, y) = color_converted;
  }
}


int __main(int argc, char** argv)
{
  if (argc < 2)
    return -1;
  const auto video_filepath = argv[1];

  omp_set_num_threads(omp_get_max_threads());

  auto video_stream = sara::VideoStream{video_filepath};

  const auto camera = make_omnidirectional_camera();

  const auto w = static_cast<int>(camera.image_sizes.x());
  const auto h = static_cast<int>(camera.image_sizes.y());

  // Mark the pixels that are in front of the camera according to the
  // mathematical model.
  const auto is_behind_camera = flag_behind_camera(camera);
  auto is_behind_camera_map = sara::Image<sara::Rgb8>{video_stream.sizes()};
  auto xu = sara::Image<float>{video_stream.sizes()};
  auto yu = sara::Image<float>{video_stream.sizes()};
  for (auto y = 0; y < h; ++y)
  {
    for (auto x = 0; x < w; ++x)
    {
      const auto ray = camera.backproject(Eigen::Vector2f(x, y));
      const auto& K = camera.K;
      const Eigen::Vector2f pu = (K * ray).hnormalized();
      if (std::abs(pu.x()) < 2e4 && std::abs(pu.y()) < 5e3)
      {
        xu(x, y) = pu.x();
        yu(x, y) = pu.y();
      }
      else
      {
        xu(x, y) = 0;
        yu(x, y) = 0;
      }
    }
  }

  auto [camera_dst, R] = make_pinhole_camera(camera);
  const auto coords_map = stereographic_projection(  //
      camera, camera_dst, R                          //
  );
  auto stereographic_projection = sara::Image<sara::Rgb8>{w, h};

  // Undistorted coordinates.
  const auto x_min = xu.flat_array().minCoeff();
  const auto y_min = yu.flat_array().minCoeff();
  const auto x_max = xu.flat_array().maxCoeff();
  const auto y_max = yu.flat_array().maxCoeff();

  const auto start = Eigen::Vector2f(x_min, y_min);
  const auto end = Eigen::Vector2f(x_max, y_max);
  const Eigen::Vector2f sizes_undistorted = end - start;
  const auto scale = std::max(sizes_undistorted.x() / float(w),
                              sizes_undistorted.y() / float(h));
  auto frame_undistorted = sara::Image<sara::Rgb8>{video_stream.sizes()};

  auto wind = sara::create_window(video_stream.sizes(),  //
                                  "Undistortable map");
  auto winu = sara::create_window(frame_undistorted.sizes(),  //
                                  "Undistorted Frame");
  auto winp = sara::create_window(stereographic_projection.sizes(),  //
                                  "Stereographic");

  while (video_stream.read())
  {
    const auto frame = video_stream.frame();

    undistort_image(frame, frame_undistorted, camera, x_min, y_min, scale);
    flag_behind_camera(frame, is_behind_camera, is_behind_camera_map);
    warp(coords_map, frame, stereographic_projection);

    sara::set_active_window(wind);
    sara::display(is_behind_camera_map);

    sara::set_active_window(winu);
    sara::display(frame_undistorted);

    sara::set_active_window(winp);
    sara::display(stereographic_projection);
  }

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
