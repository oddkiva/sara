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

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/DisjointSets/DisjointSets.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/ImageProcessing/EdgeGrouping.hpp>

#include <omp.h>


using namespace std;
using namespace DO::Sara;


constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}

constexpr long double operator"" _deg(long double x)
{
  return x * M_PI / 180;
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

  // OpenMP.
  omp_set_num_threads(omp_get_max_threads());

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  const auto downscale_factor = 2;
  auto frame_gray32f = Image<float>{};


  // Output save.
  VideoWriter video_writer{
#ifdef __APPLE__
      "/Users/david/Desktop/curve-analysis.mp4",
#else
      "/home/david/Desktop/curve-analysis.mp4",
#endif
      frame.sizes()  //
  };


  // Show the local extrema.
  create_window(frame.sizes());
  set_antialiasing();

  constexpr float high_threshold_ratio = static_cast<float>(20._percent);
  constexpr float low_threshold_ratio =
      static_cast<float>(high_threshold_ratio / 2.);
  constexpr float angular_threshold = static_cast<float>(20. / 180. * M_PI);
  const auto sigma = std::sqrt(std::pow(1.2f, 2) - 1);
#define CROP
#ifdef CROP
  const Eigen::Vector2i& p1 = {0, 0.6 * frame.height()};
  const Eigen::Vector2i& p2 = {frame.width(), 0.9 * frame.height()};
#else
  const Eigen::Vector2i& p1 = Eigen::Vector2i::Zero();
  const Eigen::Vector2i& p2 = frame.sizes();
#endif

  auto ed = EdgeDetector{{
      high_threshold_ratio,  //
      low_threshold_ratio,   //
      angular_threshold      //
  }};

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

    // Reduce our attention to the central part of the image.
    tic();
    const auto frame_cropped = crop(frame, p1, p2);
    toc("Crop");

    tic();
    frame_gray32f = frame_cropped.convert<float>();
    toc("Grayscale");

    tic();
    frame_gray32f = gaussian(frame_gray32f, sigma);
    toc("Blur");

    if (downscale_factor > 1)
    {
      tic();
      frame_gray32f = downscale(frame_gray32f, downscale_factor);
      toc("Downscale");
    }

    ed(frame_gray32f);
    auto& edges_refined = ed.pipeline.edges_simplified;

// #define SPLIT_EDGES
#ifdef SPLIT_EDGES
    tic();
    // TODO: split only if the inertias matrix is becoming isotropic.
    edges_refined = split(edges_refined, 10. * M_PI / 180.);
    toc("Edge Split");
#endif

    tic();
    auto edges = std::vector<std::vector<Eigen::Vector2d>>{};
    edges.reserve(edges_refined.size());
    for (const auto& e : edges_refined)
    {
      if (e.size() < 2)
        continue;
      if (length(e) < 10)
        continue;
      edges.emplace_back(e);
    }
    toc("Edge Filtering");


    tic();
    auto edge_stats = CurveStatistics{edges};
    toc("Edge Shape Statistics");


    const Eigen::Vector2d p1d = p1.cast<double>();
    const auto s = static_cast<float>(downscale_factor);

    auto detection = Image<Rgb8>{frame};
    detection.flat_array().fill(Black8);

    for (const auto& e : edges)
      if (e.size() >= 2)
        draw_polyline(detection, e, Blue8, p1d, s);


    display(detection);
  }

  return 0;
}
