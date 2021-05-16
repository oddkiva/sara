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

  {
    // Focal lengths in each dimension.
    const auto fx = 1063.30738864;
    const auto fy = 1064.20554291;
    // Shear component.
    const auto s = -1.00853432;
    // Principal point.
    const auto u0 = 969.55702157;
    const auto v0 = 541.26230733;

    camera_parameters.image_sizes << w, h;
    camera_parameters.K <<  //
        fx,  s, u0,  //
         0, fy, v0,  //
         0,  0,  1;
    camera_parameters.radial_distortion_coefficients <<  //
        0.50776095,
        -0.16478652;
    camera_parameters.tangential_distortion_coefficients <<  //
        0.00023093,
        0.00078712;
    camera_parameters.xi = 1.50651524;
  }

  camera_parameters.cache_inverse_calibration_matrix();

  return camera_parameters;
}


int __main(int argc, char** argv)
{
  if (argc < 2)
    return -1;
  const auto video_filepath = argv[1];

  omp_set_num_threads(omp_get_max_threads());

  auto video_stream = sara::VideoStream{video_filepath};

  const auto camera_parameters = make_omnidirectional_camera();

  const auto w = static_cast<int>(camera_parameters.image_sizes.x());
  const auto h = static_cast<int>(camera_parameters.image_sizes.y());
  const auto wh = w * h;

  // Mark the pixels that are in front of the camera according to the
  // mathematical model.
  auto is_behind_camera = sara::Image<std::uint8_t>{video_stream.sizes()};
  auto is_behind_camera_map = sara::Image<sara::Rgb8>{video_stream.sizes()};
  auto xu = sara::Image<float>{video_stream.sizes()};
  auto yu = sara::Image<float>{video_stream.sizes()};
  const auto& K = camera_parameters.K;
  for (auto y = 0; y < h; ++y)
  {
    for (auto x = 0; x < w; ++x)
    {
      const auto X = camera_parameters.backproject(Eigen::Vector2f(x, y));
      is_behind_camera(x, y) = X.z() < 0 ? 255 : 0;

      const Eigen::Vector2f pu = (K * X).hnormalized();
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

  auto video_writer = sara::VideoWriter{
      "/home/david/Desktop/undistorted-omnidirectional.mp4",  //
      {w, h},                                               //
      30                                                      //
  };

  auto video_writer2 = sara::VideoWriter{
      "/home/david/Desktop/front-behind-points-omni.mp4",  //
      {w, h},                                               //
      30                                                      //
  };

  auto i = 0;
  while (video_stream.read())
  {
    const auto frame = video_stream.frame();

#pragma omp parallel for
    for (auto p = 0; p < wh; ++p)
    {
      // Destination pixel.
      const auto y = p / w;
      const auto x = p - w * y;

      const auto x1 = x_min + x * scale;
      const auto y1 = y_min + y * scale;

      const Eigen::Vector2d xyd =
          camera_parameters.distort_v2(Eigen::Vector2f(x1, y1)).cast<double>();
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

    sara::set_active_window(wind);
    sara::display(is_behind_camera_map);

    sara::set_active_window(winu);
    sara::display(frame_undistorted);

    // video_writer.write(frame_undistorted);
    // video_writer2.write(is_behind_camera_map);
  }

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
