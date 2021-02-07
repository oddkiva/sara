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

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/DisjointSets/DisjointSets.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/Geometry/Algorithms/Polyline.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/BrownConradyCamera.hpp>
#include <DO/Sara/MultiViewGeometry/SingleView/VanishingPoint.hpp>

#include <DO/Sara/VideoIO.hpp>

#include <boost/filesystem.hpp>

#include <omp.h>


using namespace std;
using namespace DO::Sara;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}


auto initialize_camera_intrinsics_1()
{
  auto intrinsics = BrownConradyCamera<float>{};

  const auto f = 991.8030424131325;
  const auto u0 = 960;
  const auto v0 = 540;
  intrinsics.image_sizes << 1920, 1080;
  intrinsics.K << f, 0, u0, 0, f, v0, 0, 0, 1;
  intrinsics.k.setZero();
  intrinsics.p.setZero();

  return intrinsics;
}

auto initialize_camera_intrinsics_2()
{
  auto intrinsics = BrownConradyCamera<float>{};

  const auto f = 946.8984425572634;
  const auto u0 = 960;
  const auto v0 = 540;
  intrinsics.image_sizes << 1920, 1080;
  intrinsics.K << f, 0, u0, 0, f, v0, 0, 0, 1;
  intrinsics.k << -0.22996356451342749, 0.05952465745165465,
      -0.007399008111054717;
  intrinsics.p.setZero();

  return intrinsics;
}


auto initialize_crop_region_1(const Eigen::Vector2i& sizes)
{
  const Eigen::Vector2i& p1 = {0, 0.6 * sizes.y()};
  const Eigen::Vector2i& p2 = {sizes.x(), 0.9 * sizes.y()};
  return std::make_pair(p1, p2);
}

auto initialize_crop_region_2(const Eigen::Vector2i& sizes)
{
  const Eigen::Vector2i& p1 = {0.1 * sizes.x(), 0.2 * sizes.y()};
  const Eigen::Vector2i& p2 = {0.9 * sizes.x(), 0.75 * sizes.y()};
  return std::make_pair(p1, p2);
}


auto default_camera_matrix()
{
  auto P = Eigen::Matrix<float, 3, 4>{};
  P.setZero();
  P.block<3, 3>(0, 0).setIdentity();
  return P;
}


struct LocalPhotometricStatistics
{
  int patch_radius = 4;

  std::vector<Eigen::Vector2i> point_set;

  // Dense maps.
  Tensor_<float, 3> rgb;
  Tensor_<float, 3> rgb_mean;
  Tensor_<float, 3> rgb_unnormalized_std_dev;

  auto resize(int width, int height)
  {
    rgb.resize({3, height, width});
    rgb_mean.resize({3, height, width});
    rgb_unnormalized_std_dev.resize({3, height, width});
  }

  auto swap(LocalPhotometricStatistics& other)
  {
    point_set.swap(other.point_set);
    rgb.swap(other.rgb);
    rgb_mean.swap(other.rgb_mean);
    rgb_unnormalized_std_dev.swap(other.rgb_unnormalized_std_dev);
  }

  auto update_point_set(const std::vector<Eigen::Vector2i>& pts)
  {
    const auto& r = patch_radius;
    const auto w = rgb.size(2);
    const auto h = rgb.size(1);

    point_set.clear();
    point_set.reserve(pts.size());

    for (const auto& p : pts)
      if (r <= p.x() && p.x() < w - r && r <= p.y() && p.y() < h - r)
        point_set.emplace_back(p);
  }

  auto update_image(const ImageView<Rgb8>& image)
  {
    auto r_ptr = rgb[0].data();
    auto g_ptr = rgb[1].data();
    auto b_ptr = rgb[2].data();
    auto rgb_ptr = image.data();

#pragma omp parallel for
    for (auto xy = 0u; xy < image.size(); ++xy)
    {
      const auto rgb = *(rgb_ptr + xy);
      *(r_ptr + xy) = float(rgb[0]) / 255;
      *(g_ptr + xy) = float(rgb[1]) / 255;
      *(b_ptr + xy) = float(rgb[2]) / 255;
    }

#ifdef INSPECT
    for (auto i = 0; i < 3; ++i)
    {
      display(image_view(rgb[i]));
      get_key();
    }
#endif
  }

  auto calculate_mean()
  {
    // Reset the mean.
    for (auto i = 0; i < 3; ++i)
    {
      auto plane = rgb_mean[i];
      plane.flat_array().setZero();
    }

    const auto& r = patch_radius;
    const auto area = std::pow(2 * r, 2);

    auto calculate_mean_at = [&](const TensorView_<float, 2>& plane, int x,
                                 int y) {
      auto mean = float{};

      const auto xmin = x - r;
      const auto xmax = x + r;
      const auto ymin = y - r;
      const auto ymax = y + r;
      for (auto v = ymin; v < ymax; ++v)
        for (auto u = xmin; u < xmax; ++u)
          mean += plane(v, u);
      mean /= area;

      return mean;
    };

    // Loop through the points.
    for (auto i = 0; i < 3; ++i)
    {
      auto f = rgb[i];
      auto f_mean = rgb_mean[i];

#pragma omp parallel for
      for (auto i = 0u; i < point_set.size(); ++i)
      {
        const auto& p = point_set[i];
        f_mean(p.y(), p.x()) = calculate_mean_at(f, p.x(), p.y());
      }
    }

#ifdef INSPECT
    // TODO: check each plane on the display.
    auto rgb_mean_transposed = rgb_mean.transpose({1, 2, 0});
    auto rgb32f_view = ImageView<Rgb32f>(
        reinterpret_cast<Rgb32f*>(rgb_mean_transposed.data()),
        {rgb_mean_transposed.size(1), rgb_mean_transposed.size(0)});
    display(rgb32f_view);
#endif
  }

  auto calculate_unnormalized_std_dev()
  {
    // Reset the second order moment.
    for (auto i = 0; i < 3; ++i)
    {
      auto plane = rgb_unnormalized_std_dev[i];
      plane.flat_array().setZero();
    }

    const auto& r = patch_radius;
    auto calculate_unnormalized_std_dev_at =
        [&](const TensorView_<float, 2>& plane, float mean, int x, int y) {
          auto sigma = float{};

          const auto xmin = x - r;
          const auto xmax = x + r;
          const auto ymin = y - r;
          const auto ymax = y + r;
          for (auto v = ymin; v < ymax; ++v)
            for (auto u = xmin; u < xmax; ++u)
              sigma += std::pow(plane(v, u) - mean, 2);

          return sigma;
        };

    // Loop through the points.
    for (auto i = 0; i < 3; ++i)
    {
      auto f = rgb[i];
      auto f_mean = rgb_mean[i];
      auto f_sigma = rgb_unnormalized_std_dev[i];

#pragma omp parallel for
      for (auto i = 0u; i < point_set.size(); ++i)
      {
        const auto& p = point_set[i];
        const auto f_mean_xy = f_mean(p.y(), p.x());
        f_sigma(p.y(), p.x()) = calculate_unnormalized_std_dev_at(f,          //
                                                                  f_mean_xy,  //
                                                                  p.x(), p.y());
      }
    }

#ifdef INSPECT
    // TODO: check each plane on the display.
    for (auto i = 0; i < 3; ++i)
    {
      display(image_view(rgb_unnormalized_std_dev[i]));
      get_key();
    }
#endif
  }
};

struct PointMatcher {
  LocalPhotometricStatistics current;
  LocalPhotometricStatistics previous;

  auto update_statistics(const std::vector<Eigen::Vector2i>& points,
                         const ImageView<Rgb8>& frame)
  {
    current.swap(previous);

    current.update_point_set(points);
    current.update_image(frame);
    current.calculate_mean();
    current.calculate_unnormalized_std_dev();
  }
};


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
// #define CROP
#ifdef CROP
  const auto [p1, p2] = initialize_crop_region_1(frame.sizes());
#else
  const Eigen::Vector2i& p1 = Eigen::Vector2i::Zero();
  const Eigen::Vector2i& p2 = frame.sizes();
#endif

  auto ed = EdgeDetector{{
      high_threshold_ratio,  //
      low_threshold_ratio,   //
      angular_threshold      //
  }};


  // Image local statistics.
  auto statistics = LocalPhotometricStatistics{};
  statistics.resize(frame.width(), frame.height());


  // Initialize the camera matrix.
  auto intrinsics = initialize_camera_intrinsics_1();
  intrinsics.downscale_image_sizes(downscale_factor);
  SARA_CHECK(intrinsics.K);
  SARA_CHECK(intrinsics.k);
  intrinsics.calculate_K_inverse();

  auto P = default_camera_matrix();
  P = intrinsics.K * P;
  const auto Pt = P.transpose().eval();

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

#define SPLIT_EDGES
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


    auto points = std::vector<Eigen::Vector2i>{};
// #define INSPECT_ALL
#ifdef INSPECT_ALL
    points.reserve(frame.size());
    for (auto y = 0; y < frame.height(); ++y)
      for (auto x = 0; x < frame.width(); ++x)
        points.emplace_back(x, y);
#else
    for (auto e = 0u; e < edges.size(); ++e)
      for (auto p = 0u; p < edges[e].size(); ++p)
        points.emplace_back(edges[e][p].cast<int>());
#endif

    statistics.update_point_set(points);
    statistics.update_image(frame);
    statistics.calculate_mean();
    statistics.calculate_unnormalized_std_dev();

    auto detection = Image<Rgb8>{frame};
    for (const auto& e: edges)
      draw_polyline(detection, e, Red8, p1.cast<double>().eval(), downscale_factor);
    display(detection);

    // tic();
    // video_writer.write(detection);
    // toc("Video Write");
  }

  return 0;
}
