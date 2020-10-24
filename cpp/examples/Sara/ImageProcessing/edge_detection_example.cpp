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

#include <DO/Sara/FeatureDetectors/EdgeDetectionUtilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace std;
using namespace DO::Sara;


auto orientation_histograms(const ImageView<int>& edge_map,  //
                            int radius = 5, int O = 18)
{
  auto histograms = Image<float, 3>{edge_map.width(), edge_map.height(), O};
  histograms.flat_array().fill(0);

  const auto& r = radius;

// #pragma omp parallel for
  for (auto y = 0; y < edge_map.height(); ++y)
  {
    for (auto x = 0; x < edge_map.width(); ++x)
    {
      if (edge_map(x, y) < 0)
        continue;

      const auto& label = edge_map(y, x);

      // Loop over the pixels in the patch centered in p.
      for (auto v = -r; v <= r; ++v)
      {
        for (auto u = -r; u <= r; ++u)
        {
          if (u == 0 && v == 0)
            continue;

          const auto n = Eigen::Vector2i{x + u, y + v};
          if (n.x() < 0 || n.x() >= edge_map.width() ||  //
              n.y() < 0 || n.y() >= edge_map.height())
            continue;

          if  (edge_map(n) != label)
            continue;

          auto orientation = std::atan2(v, u);
          if (orientation < 0)
            orientation += M_PI;

          const auto ori_0O = orientation / M_PI * O;
          auto ori_intf = decltype(ori_0O){};
          const auto ori_frac = std::modf(ori_0O, &ori_intf);

          auto ori_0 = int(ori_intf);
          auto ori_1 = ori_0 + 1;
          if (ori_1 == O)
            ori_1 = 0;

          histograms(x, y, ori_0) += (1 - ori_frac);
          histograms(x, y, ori_1) += ori_frac;
        }
      }
    }
  }

  return histograms;
}

auto shape_statistics(const std::vector<Eigen::Vector2i>& points)
{
  auto pmat = Eigen::MatrixXf(2, points.size());
  for (auto i = 0u; i < points.size(); ++i)
    pmat.col(i) = points[i].cast<float>();

  const auto X = pmat.row(0);
  const auto Y = pmat.row(1);


  const auto x_mean = X.array().sum() / points.size();
  const auto y_mean = Y.array().sum() / points.size();

  const auto x2_mean =
      X.array().square().sum() / points.size() - std::pow(x_mean, 2);
  const auto y2_mean =
      Y.array().square().sum() / points.size() - std::pow(y_mean, 2);
  const auto xy_mean =
      (X.array() * Y.array()).sum() / points.size() - x_mean * y_mean;

  auto statistics = Eigen::Matrix<float, 5, 1>{};
  statistics << x_mean, y_mean, x2_mean, y2_mean, xy_mean;
  return statistics;
}


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

    const auto hi_thres = grad_mag.flat_array().maxCoeff() * 0.2;
    const auto lo_thres = hi_thres * 0.05;

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
  const auto skip = 1;
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

    // Group edgels by contours.
    const auto contours = connected_components(edges);
    const auto curves = connected_components(edges, grad_ori, angular_threshold);

    const auto labeled_contours = to_map(contours, edges.sizes());
    const auto labeled_curves = to_map(curves, edges.sizes());

    const auto contour_colors = random_colors(contours);
    const auto curve_colors = random_colors(curves);

    // Filter contours.
    auto is_good_contours = std::map<int, bool>{};
    for (const auto& [label, points] : contours)
    {
      auto is_good = false;
      for (const auto& p: points)
      {
        const auto& curve_id = labeled_curves(p);

        // A contour is good if it contains at least one curve.
        const auto is_curve_element = curve_id > 0;
        if (!is_curve_element)
          continue;

        // And the curve must be long enough.
        if (curves.at(curve_id).size() < 20)
          continue;

        is_good = true;
        break;
      }

      is_good_contours[label] = is_good;
    }

    // Display the good contours.
    auto contour_map = Image<Rgb8>{edges.sizes()};
    contour_map.flat_array().fill(Black8);
    // auto contour_map = frame;
    for (const auto& [label, points] : contours)
    {
      if (!is_good_contours.at(label))
        continue;

      for (const auto& p : points)
        contour_map(p) = contour_colors.at(label);
    }
    // display(contour_map);

    // Display the good contours.
    auto curve_map = Image<Rgb8>{edges.sizes()};
    curve_map.flat_array().fill(Black8);
    for (const auto& [label, points] : curves)
    {
      if (points.size() < 10)
        continue;

      for (const auto& p : points)
        curve_map(p) = curve_colors.at(label);
    }
    display(curve_map);
  }
}

GRAPHICS_MAIN()
{
  test_on_video();
  return 0;
}
