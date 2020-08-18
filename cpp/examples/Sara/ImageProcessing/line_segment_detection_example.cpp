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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Sara/Geometry/Algorithms/RobustEstimation/LineSolver.hpp>
#include <DO/Sara/Geometry/Algorithms/RobustEstimation/RANSAC.hpp>
#include <DO/Sara/Geometry/Objects/LineSegment.hpp>


using namespace std;
using namespace DO::Sara;


auto to_map(const std::map<int, std::vector<Eigen::Vector2i>>& contours,
            const Eigen::Vector2i& image_sizes)
{
  auto labeled_edges = Image<int>{image_sizes};
  labeled_edges.flat_array().fill(-1);
  for (const auto& [label, points] : contours)
  {
    for (const auto& p : points)
      labeled_edges(p) = label;
  }
  return labeled_edges;
}

auto random_colors(const std::map<int, std::vector<Eigen::Vector2i>>& contours)
{
  auto colors = std::map<int, Rgb8>{};
  for (const auto& c : contours)
    colors[c.first] = Rgb8(rand() % 255, rand() % 255, rand() % 255);
  return colors;
}

auto fit_line(const std::vector<Eigen::Vector2i>& curve_points,
              float error_threshold = 1.5f) -> std::tuple<bool, LineSegment>
{
  enum class Axis : std::uint8_t
  {
    X = 0,
    Y = 1
  };

  if (curve_points.size() < 2)
    return {false, {}};

  auto line_solver = LineSolver2D<float>{};
  auto inlier_predicate = InlierPredicate<LinePointDistance2D<float>>{
      .distance = {},                     //
      .error_threshold = error_threshold  //
  };

  auto points = Tensor_<float, 2>(curve_points.size(), 3);
  auto point_matrix = points.matrix();
  for (auto r = 0u; r < curve_points.size(); ++r)
    point_matrix.row(r) = curve_points[r]      //
                              .transpose()     //
                              .homogeneous()   //
                              .cast<float>();  //

  const auto num_iterations = std::clamp(                //
      static_cast<int>(curve_points.size() * 0.20) + 1,  //
      5, 20);
  const auto [line, inliers, subset_best] = ransac(points,            //
                                                   line_solver,       //
                                                   inlier_predicate,  //
                                                   num_iterations);

  const Eigen::Vector2f t = Projective::tangent(line).cwiseAbs();
  const auto fast_axis = t.x() > t.y() ? Axis::X : Axis::Y;

  const auto first_inlier = std::find(inliers.begin(), inliers.end(), 1);
  if (first_inlier == inliers.end())
    return {false, LineSegment{}};

  auto i = std::size_t(first_inlier - inliers.begin());

  auto tl = Eigen::Vector2f{};
  auto br = Eigen::Vector2f{};
  tl = br = point_matrix.row(i).transpose();

  for (++i; i < inliers.size(); ++i)
  {
    if (!inliers(i))
      continue;

    if (fast_axis == Axis::X)
    {
      if (tl.x() > point_matrix(i, 0))
        tl = point_matrix.row(i).hnormalized().transpose();
      if (br.x() < point_matrix(i, 0))
        br = point_matrix.row(i).hnormalized().transpose();
    }
    else
    {
      if (tl.y() > point_matrix(i, 1))
        tl = point_matrix.row(i).hnormalized().transpose();
      if (br.y() < point_matrix(i, 1))
        br = point_matrix.row(i).hnormalized().transpose();
    }
  }

  return {true, {tl.cast<double>(), br.cast<double>()}};
}

auto test_on_image()
{
  // Read an image.
  const auto image =
      imread<float>(src_path("../../../../data/sunflowerField.jpg"));

  auto sigma = sqrt(pow(1.6f, 2) - pow(0.5f, 2));
  auto image_curr = deriche_blur(image, sigma);

  constexpr auto high_threshold_ratio = 5e-2f;
  constexpr auto low_threshold_ratio = 2e-2f;
  constexpr auto angular_threshold = 20. / 180.f * M_PI;

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

    // Canny.
    const auto& grad_mag_max = grad_mag.flat_array().maxCoeff();
    const auto& high_thres = grad_mag_max * high_threshold_ratio;
    const auto& low_thres = grad_mag_max * low_threshold_ratio;

    auto edges = suppress_non_maximum_edgels(grad_mag, grad_ori,  //
                                             high_thres, low_thres);

    hysteresis(edges);

    // Extract quasi-straight curve.
    const auto curves = connected_components(edges, grad_ori, angular_threshold);
    const auto labeled_curves = to_map(curves, edges.sizes());
    const auto curve_colors = random_colors(curves);

    // Fit a line to each curve.
    auto line_segments = std::map<int, LineSegment>{};
    for (const auto& [curve_id, curve_points] : curves)
    {
      if (curve_points.size() < 5)
        continue;
      const auto [success, line_segment] = fit_line(curve_points);
      if (success)
        line_segments[curve_id] = line_segment;
    }

    // Display the fitted lines.
    fill_rect(0, 0, image_curr.width(), image_curr.height(), Black8);
    for (const auto& [curve_id, line]: line_segments)
      draw_line(line.p1(), line.p2(), curve_colors.at(curve_id), 2);

    millisleep(1);

    const auto delta = std::pow(2., 1. / 100.);
    const auto sigma = 1.6 * sqrt(pow(delta, 2 * s + 2) - pow(delta, 2 * s));
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
  // const auto video_filepath = "/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
  const auto video_filepath =
      //"/Users/david/Desktop/Datasets/videos/sample1.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample4.mp4"s;
  //     //"/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray32f = Image<float>{frame.sizes()};

  // Show the local extrema.
  create_window(frame.sizes());
  set_antialiasing();

  constexpr auto high_threshold_ratio = 5e-2f;
  constexpr auto low_threshold_ratio = 2e-2f;
  constexpr auto angular_threshold = 20. / 180.f * M_PI;

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

    frame_gray32f = frame.convert<float>();

    // Blur.
    inplace_deriche_blur(frame_gray32f, std::sqrt(std::pow(1.6f, 2) - 1));

    // Canny.
    const auto& grad = gradient(frame_gray32f);
    const auto& grad_mag = grad.cwise_transform(  //
        [](const auto& v) { return v.norm(); });
    const auto& grad_ori = grad.cwise_transform(
        [](const auto& v) { return std::atan2(v.y(), v.x()); });

    const auto& grad_mag_max = grad_mag.flat_array().maxCoeff();
    const auto& high_thres = grad_mag_max * high_threshold_ratio;
    const auto& low_thres = grad_mag_max * low_threshold_ratio;

    auto edges = suppress_non_maximum_edgels(grad_mag, grad_ori,  //
                                             high_thres, low_thres);
    hysteresis(edges);

    // Extract quasi-straight curve.
    const auto curves = connected_components(edges, grad_ori, angular_threshold);
    const auto labeled_curves = to_map(curves, edges.sizes());
    const auto curve_colors = random_colors(curves);

    // Fit a line to each curve.
    auto line_segments = std::map<int, LineSegment>{};
    for (const auto& [curve_id, curve_points] : curves)
    {
      if (curve_points.size() < 5)
        continue;
      const auto [success, line_segment] = fit_line(curve_points);
      if (success)
        line_segments[curve_id] = line_segment;
    }

    // Display the fitted lines.
    fill_rect(0, 0, frame.width(), frame.height(), Black8);
    for (const auto& [curve_id, line]: line_segments)
      draw_line(line.p1(), line.p2(), curve_colors.at(curve_id), 2);
  }
}

GRAPHICS_MAIN()
{
  // test_on_image();
  test_on_video();
  return 0;
}
