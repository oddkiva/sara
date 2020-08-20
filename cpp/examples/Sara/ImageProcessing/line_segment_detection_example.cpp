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

#include <omp.h>


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


auto fit_line_segment(const std::vector<Eigen::Vector2i>& curve_points,
                      int num_iterations,
                      bool polish = false,
                      float error_threshold = 1.5f,
                      float min_consensus_ratio = 0.5f)
    -> std::tuple<bool, LineSegment>
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

  const auto ransac_result = ransac(points,            //
                                    line_solver,       //
                                    inlier_predicate,  //
                                    num_iterations);
  const auto& line = std::get<0>(ransac_result);
  const auto& inliers = std::get<1>(ransac_result);

  // Do we have sufficiently enough inliers?
  const auto inlier_count = inliers.flat_array().count();
  if (inlier_count < min_consensus_ratio * curve_points.size())
    return {false, {}};

  auto inlier_coords = MatrixXf{inlier_count, 3};
  for (auto i = 0, j = 0; i < point_matrix.rows(); ++i)
  {
    if (!inliers(i))
      continue;

    inlier_coords.row(j) = point_matrix.row(i);
    ++j;
  }

  Eigen::Vector2f t = Projective::tangent(line).cwiseAbs();
  auto longest_axis = t.x() > t.y() ? Axis::X : Axis::Y;

  auto min_index = 0;
  auto max_index = 0;
  if (longest_axis == Axis::X)
  {
    inlier_coords.col(0).minCoeff(&min_index);
    inlier_coords.col(0).maxCoeff(&max_index);
  }
  else
  {
    inlier_coords.col(1).minCoeff(&min_index);
    inlier_coords.col(1).maxCoeff(&max_index);
  }
  Eigen::Vector2f tl = inlier_coords.row(min_index).hnormalized().transpose();
  Eigen::Vector2f br = inlier_coords.row(max_index).hnormalized().transpose();

  // Polish the line segment.
  if (polish && inlier_count > 3)
  {
    auto svd = Eigen::BDCSVD<MatrixXf>{inlier_coords, Eigen::ComputeFullU |
                                                          Eigen::ComputeFullV};
    const Eigen::Vector3f l = svd.matrixV().col(2);

    t = Projective::tangent(l).cwiseAbs();
    longest_axis = t.x() > t.y() ? Axis::X : Axis::Y;

    if (longest_axis == Axis::X)
    {
      tl.y() = -(l(0) * tl.x() + l(2)) / l(1);
      br.y() = -(l(0) * br.x() + l(2)) / l(1);
    }
    else
    {
      tl.x() = -(l(1) * tl.y() + l(2)) / l(0);
      br.x() = -(l(1) * br.y() + l(2)) / l(0);
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
      const auto num_iterations = std::clamp(                //
          static_cast<int>(curve_points.size() * 0.20) + 1,  //
          5, 20);
      const auto [success, line_segment] = fit_line_segment(curve_points,     //
                                                            num_iterations);  //
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
      //"/Users/david/Desktop/Datasets/videos/sample4.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  const auto downscale_factor = 2;
  auto frame_gray32f = Image<float>{frame.sizes() / downscale_factor};

  // Show the local extrema.
  create_window(frame.sizes() / downscale_factor);
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

    // Downscale.
    frame_gray32f = downscale(frame_gray32f, downscale_factor);

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

    auto curve_list = std::vector<std::vector<Eigen::Vector2i>>{};
    auto curve_ids = std::vector<int>{};
    for (const auto& [id, curve] : curves)
    {
      curve_list.emplace_back(curve);
      curve_ids.emplace_back(id);
    }

    // Fit a line to each curve.
    auto line_segments = std::vector<std::tuple<bool, LineSegment>>(  //
        curve_list.size(),                                            //
        {false, {}}                                                   //
    );
#pragma omp parallel for
    for (auto i = 0u; i < curve_list.size(); ++i)
    {
      const auto& curve = curve_list[i];
      if (curve.size() < 5)
        continue;

      const auto num_iterations = std::clamp(         //
          static_cast<int>(curve.size() * 0.20) + 1,  //
          5, 20);
      line_segments[i] = fit_line_segment(curve,               //
                                          num_iterations,      //
                                          /* polish */ true);  //
    }

    // Display the fitted lines.
    fill_rect(0, 0, frame_gray32f.width(), frame_gray32f.height(), Black8);
    for (auto i = 0u; i < line_segments.size(); ++i)
    {
      auto& [success, l] = line_segments[i];
      if (success)
        draw_line(l.p1(), l.p2(), curve_colors.at(curve_ids[i]), 1);
    }
  }
}


GRAPHICS_MAIN()
{
  omp_set_num_threads(omp_get_max_threads());

  // test_on_image();
  test_on_video();
  return 0;
}
