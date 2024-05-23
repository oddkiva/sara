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
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/BrownConradyDistortionModel.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <fmt/format.h>

#include <array>

#if defined(OPENMP)
#  include <omp.h>
#endif


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


auto to_map_view(const sara::BrownConradyCamera32<float>& C,
                 const sara::ImageView<sara::Rgb8>& image_plane)
    -> sara::Image<sara::Rgb8>
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
      const auto u = x - xmin_px;
      const auto v = ymax_px - y;

      const auto in_output_domain = 0 <= u && u < map_view.width() &&  //
                                    0 <= v && v < map_view.height();   //
      if (!in_output_domain)
        continue;

      // Convert the corresponding metric coordinates.
      auto xyz = Eigen::Vector3f(x / px_per_meter.value,  //
                                 camera_height.value,     //
                                 y / px_per_meter.value);

      // Project to the image.
      Eigen::Vector2d p = (C.K * xyz).hnormalized().cast<double>();

      const auto in_image_domain = 0 <= p.x() && p.x() < w - 1 &&  //
                                   0 <= p.y() && p.y() < h - 1;
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

auto make_make_conrady_camera_1()
{
  const auto f = 1305.f;
  const auto u0 = 960.f;
  const auto v0 = 540.f;
  const auto p = Eigen::Vector2f{0, 0};
  const auto k = Eigen::Vector3f{0.328456f, 0.0589776f, 0};

  auto camera_parameters = sara::BrownConradyCamera32<float>{};
  camera_parameters.image_sizes << 1920, 1080;
  camera_parameters.K << f, 0, u0, 0, f, v0, 0, 0, 1;
  camera_parameters.distortion_model.k = k;
  camera_parameters.distortion_model.p = p;

  return camera_parameters;
}

auto make_make_conrady_camera_2()
{
  const auto f = 946.898442557f;
  const auto u0 = 960.f;
  const auto v0 = 540.f;
  const auto p = Eigen::Vector2f{0, 0};
  const auto k = Eigen::Vector3f{
      -0.22996356451342749f,  //
      0.05952465745165465f,
      -0.007399008111054717f  //
  };

  auto camera_parameters = sara::BrownConradyCamera32<float>{};
  camera_parameters.image_sizes << 1920, 1080;
  camera_parameters.K << f, 0, u0, 0, f, v0, 0, 0, 1;
  camera_parameters.distortion_model.k = k;
  camera_parameters.distortion_model.p = p;

  return camera_parameters;
}

auto make_make_conrady_camera_3()
{
  const auto f = 650._px;
  const auto u0 = 640._px;
  const auto v0 = 360._px;

  auto camera_parameters = sara::BrownConradyCamera32<float>{};
  camera_parameters.K << f, 0, u0, 0, f, v0, 0, 0, 1;
  camera_parameters.distortion_model.k.setZero();
  camera_parameters.distortion_model.p.setZero();

  return camera_parameters;
}


int __main(int argc, char** argv)
{
  if (argc < 2)
    return -1;
  const auto video_filepath = argv[1];

#if defined(OPENMP)
  omp_set_num_threads(omp_get_max_threads());
#endif

  auto video_stream = sara::VideoStream{video_filepath};
  auto video_writer = sara::VideoWriter
  {
#if defined(__APPLE__)
    "/Users/david/Desktop/test.mp4",
#else
    "/home/david/Desktop/test.mp4",
#endif
        Eigen::Vector2i(map_pixel_dims[0].value, map_pixel_dims[1].value)
  };

  // one example of distortion correction.
  auto camera_parameters = make_make_conrady_camera_2();

  auto frame_undistorted = sara::Image<sara::Rgb8>{video_stream.sizes()};

  auto wu = sara::create_window(video_stream.frame().sizes(),  //
                                "Undistorted Frame");
  auto wmap = sara::create_window(                         //
      map_pixel_dims[0].value,                             //
      map_pixel_dims[1].value,                             //
      fmt::format("Bird's Eye View [Scale: {} px = 1 m]",  //
                  int(px_per_meter.value)));

  while (video_stream.read())
  {
    frame_undistorted = video_stream.frame();
    auto map_view = to_map_view(camera_parameters, video_stream.frame());

    sara::set_active_window(wu);
    sara::display(frame_undistorted);

    sara::set_active_window(wmap);
    constexpr auto interval = 20;
    for (auto i = 0; i < 5; ++i)
    {
      const auto p1 = Eigen::Vector2i(0, i * interval * px_per_meter.value);
      const auto p2 =
          Eigen::Vector2i(map_view.width(), i * interval * px_per_meter.value);
      sara::draw_line(map_view, p1.x(), p1.y(), p2.x(), p2.y(), sara::Yellow8,
                      2);
    }
    sara::display(map_view);

    // video_writer.write(map_view);
  }

  sara::close_window(wu);
  sara::close_window(wmap);

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
