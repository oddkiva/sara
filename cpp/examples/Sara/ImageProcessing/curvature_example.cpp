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

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/Curvature.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace std;
using namespace DO::Sara;


auto radial_distance(Image<float>& phi,              //
                     const Eigen::Vector2f& center,  //
                     float radius)
{
  for (auto y = 0; y < phi.height(); ++y)
  {
    for (auto x = 0; x < phi.width(); ++x)
    {
      const auto xy = Eigen::Vector2f(x, y);
      phi(x, y) = (xy - center).norm() - radius;
    }
  }
}

auto check_curvature_on_image()
{
  const auto w = 512;
  const auto h = 512;
  auto phi = Image<float, 2>(w, h);
  radial_distance(phi, Eigen::Vector2f(w, h) / 2, w / 5.);

  create_window(w, h);

  auto curvature = Image<float>{w, h};
  for (auto y = 0; y < h; ++y)
    for (auto x = 0; x < w; ++x)
      curvature(x, y) = gaussian_curvature(phi, {x, y});

  auto radius = Image<float>{w, h};
  for (auto y = 0; y < h; ++y)
    for (auto x = 0; x < w; ++x)
      radius(x, y) = 1 / curvature(x, y);

  auto c = curvature.flat_array();
  const auto cmin = curvature.flat_array().minCoeff();
  const auto cmax = curvature.flat_array().maxCoeff();
  c = (c - cmin) / (cmax - cmin);

  auto r = radius.flat_array();
  const auto rmin = curvature.flat_array().minCoeff();
  const auto rmax = curvature.flat_array().maxCoeff();
  r = (r - rmin) / (rmax - rmin);

  display(curvature);
  get_key();

  display(radius);
  get_key();
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  const auto video_filepath = argc == 2
                                  ? argv[1]
#ifdef _WIN32
                                  : "C:/Users/David/Desktop/GOPR0542.MP4"s;
#elif __APPLE__
                                  : "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
                                  : "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray32f = Image<float>{frame.sizes()};
  auto curvature = Image<float>{frame.sizes()};

  // Show the local extrema.
  create_window(frame.sizes());
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
    if (frames_read % (skip + 1) != 0)
      continue;

    std::transform(frame.begin(), frame.end(), frame_gray32f.begin(),
                   [](const auto& rgb) {
                     auto gray = float{};
                     DO::Sara::smart_convert_color(rgb, gray);
                     return gray;
                   });

    inplace_deriche_blur(frame_gray32f, 1.6f);

    for (auto y = 0; y < frame.height(); ++y)
      for (auto x = 0; x < frame.width(); ++x)
        curvature(x, y) = gaussian_curvature(frame_gray32f, {x, y});

    auto c = curvature.flat_array();
    const auto cmin = curvature.flat_array().minCoeff();
    const auto cmax = curvature.flat_array().maxCoeff();
    c = (c - cmin) / (cmax - cmin);

    // display(frame_gray32f);
    display(curvature);

    ++frames_read;
  }

  return 0;
}
