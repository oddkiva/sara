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
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/BrownConradyDistortionModel.hpp>
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
  auto intrinsics = BrownConradyCamera32<float>{};

  const auto f = 991.8030424131325f;
  const auto u0 = 960.f;
  const auto v0 = 540.f;
  intrinsics.image_sizes << 1920, 1080;
  // clang-format off
  intrinsics.K <<
    f, 0, u0,
    0, f, v0,
    0, 0,  1;
  // clang-format on
  intrinsics.distortion_model.k.setZero();
  intrinsics.distortion_model.p.setZero();

  return intrinsics;
}

auto initialize_camera_intrinsics_2()
{
  auto intrinsics = BrownConradyCamera32<float>{};

  const auto f = 946.8984425572634f;
  const auto u0 = 960.f;
  const auto v0 = 540.f;
  intrinsics.image_sizes << 1920, 1080;
  // clang-format off
  intrinsics.K <<
    f, 0, u0,
    0, f, v0,
    0, 0,  1;
  // clang-format on
  intrinsics.distortion_model.k <<
    -0.22996356451342749f,
    0.05952465745165465f,
    -0.007399008111054717f;
  intrinsics.distortion_model.p.setZero();

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


auto calculate_ncc(const Eigen::Vector2i& a, const Eigen::Vector2i& b,
                   const ImageView<float>& f, const ImageView<float>& g,
                   const ImageView<float>& f_mean,
                   const ImageView<float>& g_mean,
                   const ImageView<float>& f_sigma,
                   const ImageView<float>& g_sigma, int radius) -> float
{
  const auto a_mean = f_mean(a);
  const auto a_sigma = f_sigma(a);

  const auto b_mean = g_mean(b);
  const auto b_sigma = g_sigma(b);

  const auto& r = radius;

  // Accumulate the dot products.
  float ncc = 0.f;
  for (auto v = -r; v < r; ++v)
    for (auto u = -r; u < r; ++u)
      ncc += (f(a.x() + v, a.y() + v) - a_mean) *
             (g(b.x() + v, b.y() + u) - b_mean);

  if (std::abs(ncc) < 1e-3f)
    return 0.f;

  ncc /= a_sigma * b_sigma;
  return ncc;
}


struct LocalPhotometricStatistics
{
  int patch_radius = 8;

  std::vector<Eigen::Vector2i> point_set;

  // Dense maps.
  Image<int> point_map;
  Image<float> gray;
  Image<float> gray_mean;
  Image<float> gray_unnormalized_std_dev;

  auto resize(int width, int height)
  {
    point_map.resize(width, height);
    gray.resize(width, height);
    gray_mean.resize(width, height);
    gray_unnormalized_std_dev.resize(width, height);
  }

  auto swap(LocalPhotometricStatistics& other)
  {
    point_set.swap(other.point_set);
    point_map.swap(other.point_map);
    gray.swap(other.gray);
    gray_mean.swap(other.gray_mean);
    gray_unnormalized_std_dev.swap(other.gray_unnormalized_std_dev);
  }

  auto update_point_set(const std::vector<Eigen::Vector2i>& pts)
  {
    const auto& r = patch_radius;
    const auto w = gray.width();
    const auto h = gray.height();

    // List of point coordinates.
    point_set.clear();
    point_set.reserve(pts.size());
    for (const auto& p : pts)
    {
      if (r <= p.x() && p.x() < w - r &&  //
          r <= p.y() && p.y() < h - r)
      {
        point_set.emplace_back(p);
      }
    }

    // Point map localization.
    point_map.flat_array().fill(-1);
    for (auto i = 0u; i < point_set.size(); ++i)
      point_map(point_set[i]) = i;
  }

  auto update_image(const ImageView<float>& image)
  {
    gray = image;

// #define INSPECT_IMAGE
#ifdef INSPECT_IMAGE
    display(gray);
    get_key();
#endif
  }

  auto calculate_mean()
  {
    // Reset the mean.
    gray_mean.flat_array().setZero();

    const auto& r = patch_radius;
    const auto area = square(2 * r);

    auto calculate_mean_at = [&](const ImageView<float>& plane, int x, int y) {
      auto mean = float{};

      const auto xmin = x - r;
      const auto xmax = x + r;
      const auto ymin = y - r;
      const auto ymax = y + r;
      for (auto v = ymin; v < ymax; ++v)
        for (auto u = xmin; u < xmax; ++u)
          mean += plane(u, v);
      mean /= area;

      return mean;
    };

    // Loop through the points.
#pragma omp parallel for
    for (auto i = 0; i < static_cast<int>(point_set.size()); ++i)
    {
      const auto& p = point_set[i];
      gray_mean(p) = calculate_mean_at(gray, p.x(), p.y());
    }

// #define INSPECT_MEAN
#ifdef INSPECT_MEAN
    display(gray_mean);
    get_key();
#endif
  }

  auto calculate_unnormalized_std_dev()
  {
    // Reset the second order moment.
    gray_unnormalized_std_dev.flat_array().setZero();

    const auto& r = patch_radius;
    auto calculate_unnormalized_std_dev_at = [&](const ImageView<float>& plane,
                                                 float mean, int x, int y) {
      auto sigma = float{};

      const auto xmin = x - r;
      const auto xmax = x + r;
      const auto ymin = y - r;
      const auto ymax = y + r;
      for (auto v = ymin; v < ymax; ++v)
        for (auto u = xmin; u < xmax; ++u)
          sigma += square(plane(u, v) - mean);

      sigma = std::sqrt(sigma);

      return sigma;
    };

    // Loop through the points.
#pragma omp parallel for
    for (auto i = 0; i < static_cast<int>(point_set.size()); ++i)
    {
      const auto& p = point_set[i];
      const auto mean_xy = gray_mean(p);
      gray_unnormalized_std_dev(p) = calculate_unnormalized_std_dev_at(  //
          gray,                                                          //
          mean_xy,                                                       //
          p.x(), p.y());
    }

// #define INSPECT_SIGMA
#ifdef INSPECT_SIGMA
    // TODO: check each plane on the display.
    display(gray_unnormalized_std_dev);
    get_key();
#endif
  }
};


struct EdgeMatcher
{
  // List of edges.
  std::vector<std::vector<Eigen::Vector2d>> current_edges;
  std::vector<std::vector<Eigen::Vector2d>> previous_edges;

  // Local statistics.
  LocalPhotometricStatistics current;
  LocalPhotometricStatistics previous;

  // Radius search
  int radius = 8;

  // Matching.
  Eigen::MatrixXi candidate_matches;
  Eigen::RowVectorXi num_candidate_matches;

  auto initialize(int width, int height)
  {
    current.resize(width, height);
    previous.resize(width, height);
  }

  auto update_statistics(const std::vector<std::vector<Eigen::Vector2d>>& edges,
                         const ImageView<float>& frame)
  {
    // Store the data calculated from the previous frame.
    current_edges.swap(previous_edges);
    current.swap(previous);

    // Update the edge data.
    current_edges = edges;

    // Update the local statistics.
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

    current.update_point_set(points);
    current.update_image(frame);
    current.calculate_mean();
    current.calculate_unnormalized_std_dev();
  }

  // NCC is a sum of normalized dot products.
  auto calculate_ncc(const Eigen::Vector2i& p_prev, const Eigen::Vector2i& p_curr) const
      -> float
  {
    return ::calculate_ncc(p_prev, p_curr,
                           previous.gray, current.gray,
                           previous.gray_mean, current.gray_mean,
                           previous.gray_unnormalized_std_dev, current.gray_unnormalized_std_dev,
                           previous.patch_radius);
  }

  auto radius_search(int max_candidate_number = 10) -> void
  {
    if (previous.point_set.empty())
      return;

    SARA_DEBUG << "Radius search..." << std::endl;

    const auto w = current.gray.width();
    const auto h = current.gray.height();

    candidate_matches = -Eigen::MatrixXi::Ones(  //
        max_candidate_number,                    //
        current.point_set.size()                 //
    );
    num_candidate_matches = Eigen::RowVectorXi::Zero(current.point_set.size());

#pragma omp parallel for
    for (auto i = 0; i < static_cast<int>(current.point_set.size()); ++i)
    {
      const auto& pi = current.point_set[i];

      const auto ymin = std::max(pi.y() - radius, 0);
      const auto ymax = std::min(pi.y() + radius, h);

      const auto xmin = std::max(pi.x() - radius, 0);
      const auto xmax = std::min(pi.x() + radius, w);

      auto num_candidates = 0;
      for (auto y = ymin; y < ymax; ++y)
      {
        for (auto x = xmin; x < xmax; ++x)
        {
          const auto j = previous.point_map(x, y);
          if (j == -1)
            continue;
          candidate_matches(num_candidates, i) = j;
          ++num_candidates;

          if (num_candidates >= max_candidate_number)
            break;
        }

        if (num_candidates >= max_candidate_number)
          break;
      }
      num_candidate_matches(i) = num_candidates;
    }
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
  const auto downscale_factor = 1;
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
#endif

  auto ed = EdgeDetector{{
      high_threshold_ratio,  //
      low_threshold_ratio,   //
      angular_threshold      //
  }};


  // Initialize the camera matrix.
  auto intrinsics = initialize_camera_intrinsics_1();
  intrinsics.downscale_image_sizes(downscale_factor);

  // Edge matcher.
  auto edge_matcher = EdgeMatcher{};
  edge_matcher.initialize(frame.width() / downscale_factor,
                          frame.height() / downscale_factor);


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

#ifdef CROP
    // Reduce our attention to the central part of the image.
    tic();
    const auto frame_cropped = crop(frame, p1, p2);
    toc("Crop");
#else
    const auto& frame_cropped = frame;
#endif

    tic();
    frame_gray32f = from_rgb8_to_gray32f(frame_cropped);
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

    tic();
    edge_matcher.update_statistics(edges, frame_gray32f);
    toc("Local Photo Statistics");

    tic();
    edge_matcher.radius_search();
    toc("Radius Search");

    auto detection = Image<Rgb8>{frame};

    // Draw the previous and current of edges.
    for (const auto& e: edge_matcher.previous_edges)
      draw_polyline(detection, e, Blue8, p1.cast<double>().eval(), downscale_factor);
    for (const auto& e: edge_matcher.current_edges)
      draw_polyline(detection, e, Cyan8, p1.cast<double>().eval(), downscale_factor);


    // Draw the candidate matches.
    const auto& s = downscale_factor;
    if (edge_matcher.candidate_matches.cols() > 0)
    {
      SARA_DEBUG << "Show candidate matches" << std::endl;
      for (auto curr = 0; curr < edge_matcher.candidate_matches.cols(); ++curr)
      {
        if (edge_matcher.num_candidate_matches(curr) == 0)
          continue;

        auto& detection_copy = detection;

        const Eigen::Vector2d p_curr =                                //
            p1.cast<double>() +                                       //
            s * edge_matcher.current.point_set[curr].cast<double>();  //
#ifdef INSPECT
        SARA_CHECK(curr);
        const auto& r = edge_matcher.radius;
        draw_rect(detection_copy, p_curr.x() - s * r, p_curr.y() - s * r,
                  s * 2 * r, s * 2 * r, Cyan8, 2);
#endif

        // Draw the candidate matches.
        auto best_ncc = -1.f;
        auto best_prev = -1;
        for (auto k = 0; k < edge_matcher.num_candidate_matches(curr); ++k)
        {
          const auto& prev = edge_matcher.candidate_matches(k, curr);

          const auto ncc =
              edge_matcher.calculate_ncc(edge_matcher.previous.point_set[prev],
                                         edge_matcher.current.point_set[curr]);
          if (best_ncc < ncc)
          {
            best_ncc = ncc;
            best_prev = prev;
          }

#ifdef INSPECT
          SARA_CHECK(prev);
          SARA_CHECK(ncc);
          const Eigen::Vector2d p_prev =                                 //
              p1.cast<double>() +                                        //
              s * edge_matcher.previous.point_set[prev].cast<double>();  //
          draw_line(detection_copy, p_curr.x(), p_curr.y(), p_prev.x(),
                    p_prev.y(), Magenta8, 2);
          // display(detection_copy);
          // get_key();
#endif
        }

        if (best_ncc > 0.2)
        {
          // SARA_CHECK(curr);

          // SARA_CHECK(best_ncc);
          // SARA_CHECK(best_prev);
          const Eigen::Vector2d p_prev =                                      //
              p1.cast<double>() +                                             //
              s * edge_matcher.previous.point_set[best_prev].cast<double>();  //
          draw_line(detection_copy, p_curr.x(), p_curr.y(), p_prev.x(),
                    p_prev.y(), Yellow8, 4);
          // display(detection_copy);
        }
      }
    }
    display(detection);
    // get_key();

    // tic();
    // video_writer.write(detection);
    // toc("Video Write");
  }

  return 0;
}
