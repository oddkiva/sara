// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/BrownConradyCamera.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <array>

#include <omp.h>


using namespace std;
using namespace std::string_literals;

namespace sara = DO::Sara;
using sara::operator""_px;
using sara::operator""_m;


const auto camera_height = 1.30_m;
const auto px_per_meter = 8._px / 1._m;
const auto map_metric_dims = std::array{2.L * 10._m, 100._m};
const auto map_pixel_dims = std::array{map_metric_dims[0] * px_per_meter,
                                       map_metric_dims[1] * px_per_meter};


sara::Image<sara::Rgb8> to_map_view(const sara::BrownConradyCamera<float>& C,
                                    const sara::Image<sara::Rgb8>& image_plane)
{
  const auto xmin_px = -int(map_pixel_dims[0].value / 2);
  const auto xmax_px = int(map_pixel_dims[0].value / 2);
  const auto ymin_px = 0;
  const auto ymax_px = int(map_pixel_dims[1].value);

  const auto w = image_plane.width();
  const auto h = image_plane.height();

  auto map_view = sara::Image<sara::Rgb8>(map_pixel_dims[0].value,  //
                                          map_pixel_dims[1].value);

  for (auto y = ymin_px; y < ymax_px; ++y)
  {
    for (auto x = xmin_px; x < xmax_px; ++x)
    {
      // Convert the corresponding metric coordinates.
      auto xyz = Eigen::Vector3f{x / px_per_meter.value, camera_height.value,
                                 y / px_per_meter.value};

      // Project to the image.
      Eigen::Vector2d p = (C.K * xyz).hnormalized().cast<double>();

      const auto in_image_domain = 0 <= p.x() && p.x() < w - 1 &&  //
                                   0 <= p.y() && p.y() < h - 1;

      const auto u = x - xmin_px;
      const auto v = ymax_px - y;

      if (!in_image_domain)
      {
        map_view(u, v) = sara::Black8;
        continue;
      }

      auto color = sara::interpolate(image_plane, p);
      color /= 255;

      auto color_converted = sara::Rgb8{};
      sara::smart_convert_color(color, color_converted);

      map_view(u, v) = color_converted;
    }
  }

  return map_view;
}


int __main(int argc, char**argv)
{
  if (argc < 2)
    return -1;
  const auto video_filepath = argv[1];

  omp_set_num_threads(omp_get_max_threads());

  auto video_stream = sara::VideoStream{video_filepath};

  // one example of distortion correction.
  auto camera_parameters = sara::BrownConradyCamera<float>{};
  {
    const auto f = 946.8984425572634_px;
    const auto u0 = 960._px;
    const auto v0 = 540._px;
    const auto p = Eigen::Vector2f{0, 0};
    const auto k = Eigen::Vector3f{
        -0.22996356451342749,  //
        0.05952465745165465,
        -0.007399008111054717  //
    };
    camera_parameters.K << f, 0, u0,
                           0, f, v0,
                           0, 0,  1;
    camera_parameters.k = k;
    camera_parameters.p = p;
  }
  // // one example of distortion correction.
  // auto camera_parameters = sara::BrownConradyCamera<float>{};
  // {
  //   const auto f = 650._px;
  //   const auto u0 = 640._px;
  //   const auto v0 = 360._px;
  //   camera_parameters.K << f, 0, u0,
  //                          0, f, v0,
  //                          0, 0,  1;
  //   camera_parameters.k.setZero();
  //   camera_parameters.p.setZero();
  // }
  camera_parameters.calculate_K_inverse();
  camera_parameters.calculate_drap_lefevre_inverse_coefficients();

  auto frame_undistorted = sara::Image<sara::Rgb8>{video_stream.sizes()};
  auto frame_redistorted = sara::Image<sara::Rgb8>{video_stream.sizes()};
  auto frame_diff = sara::Image<float>{video_stream.sizes()};
  auto frame_average = sara::Image<sara::Rgb8>{video_stream.sizes()};

  auto wu = sara::create_window(video_stream.frame().sizes(),  //
                                "Undistorted Frame");
  auto wd = sara::create_window(video_stream.frame().sizes(),  //
                                "Redistorted Frame");
  auto wdiff = sara::create_window(video_stream.frame().sizes(),  //
                                   "Absolute frame diff");
  auto wavg = sara::create_window(video_stream.frame().sizes(),  //
                                  "Absolute frame average");
  auto wmap = sara::create_window(                          //
      map_pixel_dims[0].value,                              //
      map_pixel_dims[1].value,                              //
      sara::format("Bird's Eye View [Scale: %d px = 1 m]",  //
                   int(px_per_meter.value)));

  while (video_stream.read())
  {
#define DIRTY
#ifdef DIRTY
    frame_undistorted = video_stream.frame();
    const auto map_view = to_map_view(camera_parameters, video_stream.frame());
#else
    camera_parameters.undistort(video_stream.frame(), frame_undistorted);
    camera_parameters.distort_drap_lefevre(frame_undistorted, frame_redistorted);
    const auto map_view = to_map_view(camera_parameters, frame_undistorted);

    for (auto y = 0; y < frame_diff.height(); ++y)
      for (auto x = 0; x < frame_diff.width(); ++x)
        frame_diff(x, y) = (frame_redistorted(x, y).cast<float>() -
                            video_stream.frame()(x, y).cast<float>())
                               .norm();
    const auto max_val = frame_diff.flat_array().maxCoeff();
    const auto min_val = frame_diff.flat_array().minCoeff();
    frame_diff.flat_array() =
        (frame_diff.flat_array() - min_val) / (max_val - min_val);

    for (auto y = 0; y < frame_average.height(); ++y)
      for (auto x = 0; x < frame_average.width(); ++x)
      {
        const auto c1 = frame_redistorted(x, y);
        if (c1 == sara::Black8)
        {
          frame_average(x, y) = c1;
          continue;
        }

        const auto c2 = video_stream.frame()(x, y);
        const Eigen::Vector3i c = (0.5f *  //
                                   (c1.cast<float>() + c2.cast<float>()))
                                      .cast<int>();
        frame_average(x, y)(0) = c(0);
        frame_average(x, y)(1) = c(1);
        frame_average(x, y)(2) = c(2);
      }

#endif
    sara::set_active_window(wu);
    sara::display(frame_undistorted);

    sara::set_active_window(wd);
    sara::display(frame_redistorted);

    sara::set_active_window(wdiff);
    sara::display(frame_diff);

    sara::set_active_window(wavg);
    sara::display(frame_average);

    sara::set_active_window(wmap);
    sara::display(map_view);
    constexpr auto interval = 20;
    for (auto i = 0; i < 5; ++i)
      sara::draw_line(
          Eigen::Vector2i(0, i * interval * px_per_meter.value),
          Eigen::Vector2i(map_view.width(), i * interval * px_per_meter.value),
          sara::Yellow8, 2);
  }

  sara::close_window(wu);
  sara::close_window(wd);
  sara::close_window(wdiff);
  sara::close_window(wavg);
  sara::close_window(wmap);

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
