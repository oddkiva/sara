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

#include <set>

#include <omp.h>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetectionUtilities.hpp>
#include <DO/Sara/Geometry/Algorithms/RamerDouglasPeucker.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace std;
using namespace DO::Sara;


constexpr long double operator "" _percent(long double x) {
  return x / 100;
}


// ========================================================================== //
// Tests.
// ========================================================================== //
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

    const auto labeled_contours = to_map(contours, edges.sizes());
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

  constexpr auto high_threshold_ratio = 20._percent;
  constexpr auto low_threshold_ratio = high_threshold_ratio / 2.;
  constexpr auto angular_threshold = 20. / 180.f * M_PI;
  const auto sigma = std::sqrt(std::pow(1.6f, 2) - 1);
  const Eigen::Vector2i& p1 = Eigen::Vector2i::Zero();
  const Eigen::Vector2i& p2 = frame.sizes();

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

    tic();
    const auto& grad = gradient(frame_gray32f);
    toc("Gradient");

    tic();
    const auto& grad_mag = grad.cwise_transform(  //
        [](const auto& v) { return v.norm(); });
    const auto& grad_ori = grad.cwise_transform(
        [](const auto& v) { return std::atan2(v.y(), v.x()); });
    toc("Polar Coordinates");

    tic();
    const auto& grad_mag_max = grad_mag.flat_array().maxCoeff();
    const auto& high_thres = grad_mag_max * high_threshold_ratio;
    const auto& low_thres = grad_mag_max * low_threshold_ratio;

    auto edgels = suppress_non_maximum_edgels(grad_mag, grad_ori,  //
                                              high_thres, low_thres);
    toc("Thresholding");


#define SIMULTANEOUS_HYSTERESIS_AND_GROUPING
#ifdef SIMULTANEOUS_HYSTERESIS_AND_GROUPING
    tic();
    const auto edges = perform_hysteresis_and_grouping(edgels,    //
                                                       grad_ori,  //
                                                       angular_threshold);
    toc("Simultaneous Hysteresis & Edge Grouping");
#else
    tic();
    hysteresis(edgels);
    const auto edges = connected_components(edgels,    //
                                            grad_ori,  //
                                            angular_threshold);
    // const auto edges = connected_components(edgels);
    toc("Hysteresis then Edge Grouping");
#endif

    tic();
    auto edges_as_list =
        std::vector<std::vector<Eigen::Vector2i>>(edges.size());
    std::transform(edges.begin(), edges.end(), edges_as_list.begin(),
                   [](const auto& e) { return e.second; });
    toc("To vector");

    tic();
    auto edges_simplified =
        std::vector<std::vector<Eigen::Vector2d>>(edges_as_list.size());
#pragma omp parallel for
    for (auto i = 0u; i < edges_as_list.size(); ++i)
    {
      const auto& edge = extract_longest_curve(edges_as_list[i]);

      auto edges_converted = std::vector<Eigen::Vector2d>(edge.size());
      std::transform(edge.begin(), edge.end(), edges_converted.begin(),
                     [](const auto& p) { return p.template cast<double>(); });

      edges_simplified[i] = ramer_douglas_peucker(edges_converted, 1.);
    }
    toc("Longest Curve Extraction & Simplification");

#ifdef REFINE_EDGES
    tic();
    auto edges_refined = edges_simplified;
#  pragma omp parallel for
    for (auto i = 0u; i < edges_refined.size(); ++i)
      for (auto& p : edges_refined[i])
        p = refine(grad_mag, p.cast<int>()).cast<double>();
    toc("Refine Edge Localisation");
#else
    auto edges_refined = edges_simplified;
#endif


    // tic();
    // edges_refined = split(edges_refined);
    // toc("Edge Split");


    // Display the quasi-straight edges.
    tic();
    auto edge_colors = std::vector<Rgb8>(edges_refined.size());
    for (auto& c : edge_colors)
      c << rand() % 255, rand() % 255, rand() % 255;

    auto detection = frame;
    const Eigen::Vector2d p1d = p1.cast<double>();
#pragma omp parallel for
    for (auto e = 0u; e < edges_refined.size(); ++e)
    {
      // const auto& edge = edges_simplified[e];
      const auto& edge_refined = edges_refined[e];
      if (edge_refined.size() < 2)
        continue;

      const auto& color = edge_colors[e];
      const auto& s = downscale_factor;
      // draw_polyline(detection, edge, White8, p1d, s);
      draw_polyline(detection, edge_refined, color, p1d, s);
    }
    display(detection);
    toc("Draw");

    // get_key();
  }
}


GRAPHICS_MAIN()
{
  omp_set_num_threads(omp_get_max_threads());
  test_on_video();
  return 0;
}


#ifdef DETECT_JUNCTIONS
// Detect junctions.
tic();
SARA_DEBUG << "Orientation histograms..." << std::endl;
const auto ori_hists = orientation_histograms(edgels, grad_ori,   //
                                              /* num_bins */ 36,  //
                                              /* radius */ 3);
SARA_DEBUG << "Orientation peaks..." << std::endl;
const auto ori_peaks = peaks(ori_hists);
SARA_DEBUG << "Orientation peak counts..." << std::endl;
const auto ori_peak_counts = peak_counts(ori_peaks);
#endif
