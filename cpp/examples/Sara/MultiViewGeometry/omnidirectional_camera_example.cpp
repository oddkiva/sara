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
    const auto f = 677.3246133600308;
    const auto u0 = 960;
    const auto v0 = 540;

    camera_parameters.image_sizes << w, h;
    camera_parameters.K <<  //
        f, 0, u0,  //
        0, f, v0,  //
        0, 0,  1;
    camera_parameters.radial_distortion_coefficients <<
        -0.20f, 0.1321295087447987f;
    camera_parameters.tangential_distortion_coefficients <<
        -0.06844064024539671f, 0.01237548905484928f;
    camera_parameters.xi = 0.8f;
  }

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

  const auto offset = 1000;
  const auto start = Eigen::Vector2i(-offset, -offset);
  const auto end = Eigen::Vector2i(w + offset, h + offset);
  auto frame_undistorted = sara::Image<sara::Rgb8>{end - start};

  const auto wu = frame_undistorted.width();
  const auto hu = frame_undistorted.height();
  const auto whu = wu * hu;

  auto frame_undistortable = sara::Image<sara::Rgb8>{video_stream.sizes()};

  // Mark the pixels that can be undistorted because of the limitations of the
  // mathematical model.
  auto undistortable_map = sara::Image<std::uint8_t>{video_stream.sizes()};
  for (auto y = 0; y < h; ++y)
  {
    for (auto x = 0; x < w; ++x)
    {
      const auto pu = camera_parameters.undistort_v2(Eigen::Vector2f(x, y));
      const auto undistortable = !std::isnan(pu.x()) && !std::isnan(pu.y());
      undistortable_map(x, y) = undistortable ? 255 : 0;
    }
  }

  auto wind = sara::create_window(video_stream.sizes(),  //
                                  "Undistortable map");
  auto winu = sara::create_window(frame_undistorted.sizes(),  //
                                  "Undistorted Frame");

  auto video_writer = sara::VideoWriter{
      "/home/david/Desktop/undistorted-omnidirectional.mp4",  //
      {wu, hu},                                       //
      30                                              //
  };

  while (video_stream.read())
  {
    const auto frame = video_stream.frame();

#pragma omp parallel for
    for (auto p = 0; p < whu; ++p)
    {
      // Destination pixel.
      const auto y = p / wu;
      const auto x = p - wu * y;

      const auto y1 = y + start.y();
      const auto x1 = x + start.x();

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

      frame_undistortable(x, y) = undistortable_map(x, y) != 0  //
                                      ? frame(x, y)
                                      : sara::Black8;
    }

    sara::set_active_window(wind);
    sara::display(frame_undistortable);

    sara::set_active_window(winu);
    sara::display(frame_undistorted);

    video_writer.write(frame_undistorted);
  }

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
