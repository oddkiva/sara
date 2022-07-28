// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/EdgeGrouping.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <boost/filesystem.hpp>

#include <omp.h>


using namespace std;
using namespace DO::Sara;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
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

  // Parse command line.
  const auto video_filepath = argc >= 2
                                  ? argv[1]
#ifdef _WIN32
                                  : "C:/Users/David/Desktop/IMG_1895.MOV"s;
#elif __APPLE__
                                  : "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
                                  : "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif
  const auto downscale_factor = argc >= 3 ? std::stof(argv[2]) : 1.f;
  const auto skip = argc >= 4 ? std::stoi(argv[3]) : 0;
  const auto sigma = argc >= 5 ? std::stof(argv[4]) : 1.f;

  // OpenMP.
  omp_set_num_threads(omp_get_max_threads());

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray32f = Image<float>{frame.sizes()};

  const Eigen::Vector2i image_ds_sizes =
      (frame.sizes().cast<float>() / downscale_factor)
          .array()
          .round()
          .matrix()
          .cast<int>();
  auto frame_gray32f_ds = Image<float>{image_ds_sizes};

  // Output save.
  namespace fs = boost::filesystem;
  const auto basename = fs::basename(video_filepath);
  VideoWriter video_writer{
#ifdef _WIN32
      "C:/Users/David/Desktop/" + basename + ".edge-detection.mp4",
#elif __APPLE__
      "/Users/david/Desktop/" + basename + ".edge-detection.mp4",
#else
      "/home/david/Desktop/" + basename + ".edge-detection.mp4",
#endif
      frame.sizes()  //
  };

  // Show the local extrema.
  create_window(frame.sizes());
  set_antialiasing();

  constexpr float high_threshold_ratio = static_cast<float>(4._percent);
  constexpr float low_threshold_ratio =
      static_cast<float>(high_threshold_ratio / 2.);
  constexpr float angular_threshold = static_cast<float>((10._deg).value);

  auto ed = EdgeDetector{{
      high_threshold_ratio,  //
      low_threshold_ratio,   //
      angular_threshold      //
  }};

  auto frames_read = 0;
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
    SARA_DEBUG << "Processing frame " << frames_read << std::endl;

    tic();
    frame_gray32f = from_rgb8_to_gray32f(frame);
    toc("Grayscale");

    tic();
    frame_gray32f = gaussian(frame_gray32f, sigma);
    toc("Blur");

    tic();
    resize_v2(frame_gray32f, frame_gray32f_ds);
    toc("Downscale");

    ed(frame_gray32f_ds);
    const auto& edges_simplified = ed.pipeline.edges_simplified;

    tic();
    auto disp = frame.convert<float>().convert<Rgb8>();
    for (const auto& e : edges_simplified)
    {
      if (e.size() >= 2 && length(e) > 10)
      {
        const auto color = Rgb8(rand() % 255, rand() % 255, rand() % 255);
        draw_polyline(disp, e, color, Eigen::Vector2d{0, 0}, downscale_factor);
      }
    }
    display(disp);
    toc("Display");
    get_key();

    tic();
    video_writer.write(frame);
    toc("Video Write");
  }

  return 0;
}
