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
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <omp.h>


using namespace std;
using namespace DO::Sara;


constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}


// TODO:
// Rectangular approximation.
// - linear direction mean
// - aligned density


auto test_on_image()
{
  // Read an image.
  const auto image =
      imread<float>(src_path("../../../../data/sunflowerField.jpg"));

  auto sigma = sqrt(pow(1.6f, 2) - pow(0.5f, 2));
  auto image_curr = deriche_blur(image, sigma);

  create_window(image.sizes());
  display(image_curr);

  for (auto s = 0; s < 500; ++s)
  {
    // Blur.
    const auto grad = gradient(image_curr);
    const auto grad_mag = grad.cwise_transform(  //
        [](const auto& v) { return v.norm(); });
    const auto grad_ori = grad.cwise_transform(
        [](const auto& v) { return std::atan2(v.y(), v.x()); });

    const auto hi_thres = grad_mag.flat_array().maxCoeff() * 0.2f;
    const auto lo_thres = hi_thres * 0.05f;

    auto edges = suppress_non_maximum_edgels(grad_mag, grad_ori,  //
                                             hi_thres, lo_thres);

    hysteresis(edges);

    const auto contours = connected_components(edges);

    const auto labeled_contours = to_dense_map(contours, edges.sizes());
    const auto colors = random_colors(contours);

    auto contour_map = Image<Rgb8>{edges.sizes()};
    contour_map.flat_array().fill(Black8);
    for (const auto& [label, points] : contours)
      for (const auto& p : points)
        contour_map(p) = colors.at(label);
    display(contour_map);

    millisleep(1);

    const auto delta = std::pow(2.f, 1 / 100.f);
    const auto sigma = 1.6f * sqrt(pow(delta, 2 * s + 2) - pow(delta, 2 * s));
    image_curr = deriche_blur(image_curr, sigma);
  }

  get_key();
}

auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath =
      //"/Users/david/Desktop/Datasets/videos/sample1.mp4"s;
      //"/Users/david/Desktop/Datasets/videos/sample4.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  // const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
  // const auto video_filepath = "/home/david/Desktop/Datasets/ha/text_5.avi"s;
  // const auto video_filepath =
  // "/home/david/Desktop/Datasets/ha/GH010175_converted.mp4"s;
  const auto video_filepath = "/home/david/Desktop/Datasets/ha/turn_bikes.mp4"s;
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  const auto downscale_factor = 2;
  auto frame_gray32f = Image<float>{};

  // Show the local extrema.
  create_window(frame.sizes());
  set_antialiasing();

  constexpr float high_threshold_ratio = 20._percent;
  constexpr float low_threshold_ratio = high_threshold_ratio / 2.;
  constexpr float angular_threshold = 20. / 180. * M_PI;
  const auto sigma = std::sqrt(std::pow(1.6f, 2) - 1);
  const Eigen::Vector2i& p1 = frame.sizes() / 4;  // Eigen::Vector2i::Zero();
  const Eigen::Vector2i& p2 = frame.sizes() * 3 / 4;  // frame.sizes();

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
    auto edges_refined = ed.pipeline.edges_simplified;

    tic();
    edges_refined = split(edges_refined, 10. * M_PI / 180.);
    toc("Edge Split");

    tic();
    auto line_segments =
        std::vector<LineSegment>(edges_refined.size(), {{0., 0.}, {0., 0.}});
    for (auto i = 0u; i < edges_refined.size(); ++i)
      if (edges_refined[i].size() >= 2)
        // line_segments[i] = fit_line_segment(edges_refined[i]);
        line_segments[i] = {edges_refined[i].front(), edges_refined[i].back()};
    toc("Line Segment Fitting");

    // Display the quasi-straight edges.
    tic();
    auto edge_colors = std::vector<Rgb8>(edges_refined.size(), Red8);
#ifdef RANDOM_COLORS
    for (auto& c : edge_colors)
      c << rand() % 255, rand() % 255, rand() % 255;
#endif

    auto detection = frame;
    const Eigen::Vector2d p1d = p1.cast<double>();
    const auto& s = downscale_factor;

#pragma omp parallel for
    for (auto e = 0u; e < edges_refined.size(); ++e)
    {
      // const auto& edge = edges_simplified[e];
      const auto& edge_refined = edges_refined[e];
      if (edge_refined.size() < 2)
        continue;

      const auto& color = edge_colors[e];
      draw_polyline(detection, edge_refined, color, p1d, s);

      if (edge_refined.size() == 2)
      {
        const Point2d a = p1d + s * line_segments[e].p1();
        const Point2d b = p1d + s * line_segments[e].p2();
        draw_line(detection, a.x(), a.y(), b.x(), b.y(),  //
                  Green8,                                 //
                  /* line_width */ 2);
      }
    }
    display(detection);

    toc("Draw");
  }
}


GRAPHICS_MAIN()
{
  omp_set_num_threads(omp_get_max_threads());
  test_on_video();
  return 0;
}
