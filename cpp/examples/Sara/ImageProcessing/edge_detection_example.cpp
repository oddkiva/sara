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

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetectionUtilities.hpp>
#include <DO/Sara/Geometry/Algorithms/RamerDouglasPeucker.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace std;
using namespace DO::Sara;


//! TODO: see if reweighting with edge length improves the split.
template <typename Point>
auto split(const std::vector<Point>& ordered_points,
           const float angle_threshold = M_PI / 6)
    -> std::vector<std::vector<Point>>
{
  if (angle_threshold >= M_PI)
    throw std::runtime_error{"Invalid angle threshold!"};

  if (ordered_points.size() < 3)
    return {ordered_points};

  const auto& p = ordered_points;
  const auto cos_threshold = std::cos(angle_threshold);

  auto deltas = std::vector<Point>(p.size() - 1);
  for (auto i = 0u; i < deltas.size(); ++i)
    deltas[i] = (p[i + 1] - p[i]).normalized();

  auto cuts = std::vector<std::uint8_t>(p.size(), 0);
  for (auto i = 0u; i < deltas.size() - 1; ++i)
  {
    const auto cosine = deltas[i].dot(deltas[i + 1]);
    const auto cut_at_i = cosine < cos_threshold;
    cuts[i + 1] = cut_at_i;
  }

  auto pp = std::vector<std::vector<Point>>{};
  pp.push_back({p[0]});
  for (auto i = 1u; i < p.size(); ++i)
  {
    pp.back().push_back(p[i]);
    if (cuts[i] == 1)
      pp.push_back({p[i]});
  }

  return pp;
}


template <typename Point>
auto split(const std::vector<std::vector<Point>>& edges,
                  const float angle_threshold = M_PI / 6)
    -> std::vector<std::vector<Point>>
{
  auto edges_split = std::vector<std::vector<Point>>{};
  edges_split.reserve(2 * edges.size());
  for (const auto& e: edges)
    append(edges_split, split(e, angle_threshold));

  return edges_split;
}

// inline auto fit_line_segment(const std::vector<Eigen::Vector2i>& points)
//     -> LineSegment
// {
//   auto coords = MatrixXf{points.size(), 3};
//   for (auto i = 0; i < coords.rows(); ++i)
//     coords.row(i) = points[i].homogeneous().transpose().cast<float>();
//
//   // Calculate the line equation `l`.
//   auto svd = Eigen::BDCSVD<MatrixXf>{coords,
//                                      Eigen::ComputeFullU | Eigen::ComputeFullV};
//
//   const Eigen::Vector3f l = svd.matrixV().col(2);
//
//   // Direction vector.
//   const auto t = Projective::tangent(l).cwiseAbs();
//
//   enum class Axis : std::uint8_t
//   {
//     X = 0,
//     Y = 1
//   };
//   const auto longest_axis = t.x() > t.y() ? Axis::X : Axis::Y;
//
//   Eigen::Vector2d p1 = points.front().cast<float>();
//   Eigen::Vector2d p2 = points.back().cast<float>();
//
//   if (longest_axis == Axis::X)
//   {
//     p1.y() = -(l(0) * p1.x() + l(2)) / l(1);
//     p2.y() = -(l(0) * p2.x() + l(2)) / l(1);
//   }
//   else
//   {
//     p1.x() = -(l(1) * p1.y() + l(2)) / l(0);
//     p2.x() = -(l(1) * p2.y() + l(2)) / l(0);
//   }
//
//   return {p1.cast<double>(), p2.cast<double>()};
// }


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

  constexpr auto high_threshold_ratio = 15e-2f;
  constexpr auto low_threshold_ratio = 8e-2f;
  constexpr auto angular_threshold = 20. / 180.f * M_PI;
  const auto sigma = std::sqrt(std::pow(1.6f, 2) - 1);
  const Eigen::Vector2i& p1 = Eigen::Vector2i::Zero();
  const Eigen::Vector2i& p2 = frame.sizes();

  auto frames_read = 0;
  const auto skip = 2;
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
    // const auto edges = connected_components(edgels,    //
    //                                         grad_ori,  //
    //                                         angular_threshold);
    const auto edges = connected_components(edgels);
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

      edges_simplified[i] = ramer_douglas_peucker(edges_converted, 2.);
    }
    toc("Longest Curve Extraction & Simplification");

    // tic();
    // edges_simplified = split(edges_simplified);
    // toc("Edge Split");


    // Display the quasi-straight edges.
    tic();
    auto edge_colors = std::vector<Rgb8>(edges_simplified.size());
    for (auto& c : edge_colors)
      c << rand() % 255, rand() % 255, rand() % 255;

    auto detection = frame;
#pragma omp parallel for
    for (auto e = 0u; e < edges_simplified.size(); ++e)
    {
      const auto& edge = edges_simplified[e];
      if (edge.size() < 2)
        continue;

      const auto& color = edge_colors[e];
      const auto& s = downscale_factor;
      for (auto i = 0u; i < edge.size() - 1; ++i)
      {
        const auto& a = p1.cast<double>() + s * edge[i];
        const auto& b = p1.cast<double>() + s * edge[i + 1];
        draw_line(detection, a.x(), a.y(), b.x(), b.y(), color, 1, true);
        fill_circle(detection, a.x(), a.y(), 2, color);
        if (i == edge.size() - 2)
          fill_circle(detection, b.x(), b.y(), 2, color);
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


#ifdef DEBUG
    tic();
    const auto& edge = edges_simplified.front();
    const auto& splits = split(edge, M_PI / 10);

    display(frame);

    // Check the splits.
    if (edge.size() >= 3)
    {
      const auto color = Red8;
      const auto& s = downscale_factor;
      for (auto i = 0u; i < edge.size() - 1; ++i)
      {
        const auto& a = p1.cast<double>() + s * edge[i];
        const auto& b = p1.cast<double>() + s * edge[i + 1];
        draw_line(a.x(), a.y(), b.x(), b.y(), color, 1);
        fill_circle(a.x(), a.y(), 2, color);
        if (i == edge.size() - 2)
          fill_circle(b.x(), b.y(), 2, color);
      }
      get_key();

      for (const auto& e : splits)
      {
        const auto color = Green8;
        for (auto i = 0u; i < e.size() - 1; ++i)
        {
          const auto& a = p1.cast<double>() + s * e[i];
          const auto& b = p1.cast<double>() + s * e[i + 1];
          draw_line(a.x(), a.y(), b.x(), b.y(), color, 1);
          fill_circle(a.x(), a.y(), 2, color);
          if (i == e.size() - 2)
            fill_circle(b.x(), b.y(), 2, color);
        }
        get_key();
      }
    }
    toc("Edge Split");
#endif
