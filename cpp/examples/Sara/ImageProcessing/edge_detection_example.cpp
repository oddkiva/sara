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
#include <DO/Sara/MultiViewGeometry/Camera/BrownConradyCamera.hpp>
#include <DO/Sara/MultiViewGeometry/SingleView/VanishingPoint.hpp>

#include <DO/Sara/VideoIO.hpp>

#include <drafts/ImageProcessing/EdgeGrouping.hpp>

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
  auto frame_undistorted = Image<Rgb8>{video_stream.sizes()};
  const auto downscale_factor = 2;
  auto frame_gray32f = Image<float>{};


  // Output save.
  namespace fs = boost::filesystem;
  const auto basename = fs::basename(video_filepath);
  VideoWriter video_writer{
#ifdef __APPLE__
      "/Users/david/Desktop/" + basename + ".curve-analysis.mp4",
#else
      "/home/david/Desktop/" + basename + ".curve-analysis.mp4",
#endif
      frame.sizes()  //
  };


  // Show the local extrema.
  create_window(frame.sizes());
  set_antialiasing();

  constexpr float high_threshold_ratio = static_cast<float>(20._percent);
  constexpr float low_threshold_ratio =
      static_cast<float>(high_threshold_ratio / 2.);
  constexpr float angular_threshold = static_cast<float>((20._deg).value);
  const auto sigma = std::sqrt(std::pow(1.2f, 2) - 1);

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
    SARA_DEBUG << "Processing frame " << frames_read << std::endl;

    tic();
    frame_gray32f = frame.convert<float>();
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
    auto edges = ed.pipeline.edges_simplified;

    tic();
    // TODO: split only if the inertias matrix is becoming isotropic.
    edges = split(edges, 10. * M_PI / 180.);
    toc("Edge Split");

    tic();
    const auto edge_stats = CurveStatistics{edges};
    toc("Edge Shape Statistics");

    tic();
    auto line_segments = extract_line_segments_quick_and_dirty(edge_stats);
    {
      auto line_segments_filtered = std::vector<LineSegment>{};
      line_segments_filtered.reserve(line_segments.size());

      for (const auto& s : line_segments)
        if (s.length() > 10)
          line_segments_filtered.emplace_back(s);

      line_segments.swap(line_segments_filtered);
    }

    // Go back to the original pixel coordinates.
    const auto s = static_cast<float>(downscale_factor);
    for (auto& ls: line_segments)
    {
      ls.p1() *= s;
      ls.p2() *= s;
    }
    const auto lines = to_lines(line_segments);
    toc("Line Segment Extraction");

    // Draw the detected line segments.
    for (const auto& s : line_segments)
      draw_line(frame, s.x1(), s.y1(), s.x2(), s.y2(), Red8, 2);
    display(frame);


    tic();
    video_writer.write(frame);
    toc("Video Write");
  }

  return 0;
}
