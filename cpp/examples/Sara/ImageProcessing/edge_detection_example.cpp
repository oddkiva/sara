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
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace std;
using namespace DO::Sara;


// ========================================================================== //
// Utilities.
// ========================================================================== //
template <typename T>
auto find_peaks(
    const Map<const Eigen::Array<T, Eigen::Dynamic, 1>>& circular_data,  //
    Map<Eigen::Array<std::uint8_t, Eigen::Dynamic, 1>>& peaks,           //
    T peak_ratio_thres = static_cast<T>(0.8))                            //
    -> void
{
  if (circular_data.size() == 0)
    return;

  const auto& max = circular_data.maxCoeff();
  const auto& N = circular_data.size();

  for (auto i = Eigen::Index{}; i < N; ++i)
  {
    const auto prev = i > 0 ? i - 1 : N;
    const auto next = i < N - 1 ? i + 1 : 0;
    peaks[i] = circular_data(i) >= peak_ratio_thres * max &&
               circular_data(i) > circular_data(prev) &&
               circular_data(i) > circular_data(next);
  }
}

template <typename T>
auto peak_residual(
    const Map<const Eigen::Array<T, Eigen::Dynamic, 1>>& circular_data,  //
    Eigen::Index i)                                                      //
    -> T
{
  const auto& N = circular_data.size();

  const auto prev = i > 0 ? i - 1 : N;
  const auto next = i < N - 1 ? i + 1 : 0;

  const auto& y0 = circular_data(prev);
  const auto& y1 = circular_data(i);
  const auto& y2 = circular_data(next);

  const auto fprime = (y2 - y0) / 2.f;
  const auto fsecond = y0 - 2.f * y1 + y2;

  const auto h = -fprime / fsecond;

  return T(i) + T(0.5) + h;
}


// ========================================================================== //
// Operation on tensors.
// ========================================================================== //
auto orientation_histograms(const ImageView<std::uint8_t>& edge_map,  //
                            const ImageView<float>& orientation_map,  //
                            int num_bins = 18, int radius = 5)
{
  auto histograms =
      Tensor_<float, 3>{edge_map.height(), edge_map.width(), num_bins};
  histograms.flat_array().fill(0);

  const auto& r = radius;

#pragma omp parallel for
  for (auto y = 0; y < edge_map.height(); ++y)
  {
    for (auto x = 0; x < edge_map.width(); ++x)
    {
      if (edge_map(x, y) == 0)
        continue;

      const auto& label = edge_map(x, y);

      // Loop over the pixels in the patch centered in p.
      for (auto v = -r; v <= r; ++v)
      {
        for (auto u = -r; u <= r; ++u)
        {
          // Boundary conditions.
          const auto n = Eigen::Vector2i{x + u, y + v};
          if (n.x() < 0 || n.x() >= edge_map.width() ||  //
              n.y() < 0 || n.y() >= edge_map.height())
            continue;

          // Only consider neighbors with the same label.
          if  (edge_map(n) != label)
            continue;

          const auto& orientation = std::abs(orientation_map(n) - orientation_map(x, y));

          auto ori_0O = static_cast<float>(orientation / (2 * M_PI) * num_bins);
          if (ori_0O >= num_bins)
            ori_0O -= num_bins;
          auto ori_intf = decltype(ori_0O){};
          const auto ori_frac = std::modf(ori_0O, &ori_intf);

          auto ori_0 = int(ori_intf);
          auto ori_1 = ori_0 + 1;
          if (ori_1 == num_bins)
            ori_1 = 0;

          histograms(y, x, ori_0) += (1 - ori_frac);
          histograms(y, x, ori_1) += ori_frac;
        }
      }
    }
  }

  return histograms;
}

auto peaks(const TensorView_<float, 3>& histograms,
           float peak_ratio_thres = 0.8f)
{
  auto peaks = Tensor_<std::uint8_t, 3>{histograms.sizes()};

#pragma omp parallel for
  for (auto y = 0; y < histograms.size(0); ++y)
  {
    for (auto x = 0; x < histograms.size(1); ++x)
    {
      const auto& h = histograms[y][x].flat_array();
      auto p = peaks[y][x].flat_array();
      find_peaks(h, p, peak_ratio_thres);
    }
  }

  return peaks;
}

auto peak_counts(const TensorView_<std::uint8_t, 3>& peaks)
{
  const Eigen::Vector2i num_peaks_sizes = peaks.sizes().head(2);
  auto peak_counts = Tensor_<std::uint8_t, 2>{num_peaks_sizes};

#pragma omp parallel for
  for (auto y = 0; y < peak_counts.size(0); ++y)
    for (auto x = 0; x < peak_counts.size(1); ++x)
      peak_counts(y, x) = peaks[y][x].flat_array().count();

  return peak_counts;
}

auto peak_residuals(const TensorView_<float, 3>& circular_data,
                    const TensorView_<std::uint8_t, 3>& peak_indices)
{
  auto peak_residuals = Tensor_<float, 3>{circular_data.sizes()};

#pragma omp parallel for
  for (auto y = 0; y < circular_data.size(0); ++y)
  {
    for (auto x = 0; x < circular_data.size(1); ++x)
    {
      const auto& h = circular_data[y][x].flat_array();
      auto p = peak_indices[y][x].flat_array();

      for (auto i = 0; i < p.size(); ++i)
        peak_residuals(y, x, i) = p[i] == 0 ? 0 : peak_residual(h, i);
    }
  }

  return peak_residuals;
}


// ========================================================================== //
// Edge statistics.
// ========================================================================== //
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


// LINE SEGMENT DETECTION.
// When connecting pixel, also update the statistics of tne line.
// - orientation (cf. §2.5 in LSD)
// - rectangular approximation (§2.6)
// - density of aligned points (§2.8)

struct EdgeStatistics {
  Eigen::Vector2f orientation_sum = Eigen::Vector2f::Zero();
  float total_mass = {};
  Eigen::Vector2f unnormalized_center = Eigen::Vector2f::Zero();
  Eigen::Vector2f unnormalized_inertia = Eigen::Vector2f::Zero();

  std::vector<Eigen::Vector2i> points;

  auto angle() const -> float
  {
    return std::atan2(orientation_sum.y(), orientation_sum.x());
  }

  auto center() const -> Eigen::Vector2f
  {
    return unnormalized_center / total_mass;
  }

  auto inertia() const -> Eigen::Vector3f {
    const auto& c = center();
    const auto& m = unnormalized_inertia;
    auto m1 = Eigen::Vector3f{};
    m1(0) = m(0) * m(0) - 2 * c.x() * m(0) + c.x() * c.x();
    m1(1) = m(1) * m(1) - 2 * c.y() * m(1) + c.y() * c.y();
    m1(2) = m(0) * m(1) - c.x() * m(1) - m(0) * c.y() + c.x() * c.y();
    m1 /=  total_mass;
    return m1;
  }

  auto rectangle() const -> float {
    return {};
  }

  auto density() const -> float;
};

struct CoordsValue
{
  Eigen::Vector2i coords;
  float value;

  auto operator<(const CoordsValue& other) const
  {
    return value > other.value;
  }
};


inline auto group_line_segments(const ImageView<std::uint8_t>& edges,
                                const ImageView<float>& mag,
                                const ImageView<float>& ori,
                                float angular_threshold)
{
  const auto index = [&edges](const Eigen::Vector2i& p) {
    return p.y() * edges.width() + p.x();
  };

  const auto is_edgel = [&edges](const Eigen::Vector2i& p) {
    return edges(p) == 255;
  };

  const auto vec = [](float o) {
    return Eigen::Vector2f{cos(o), sin(o)};
  };

  const auto angular_distance = [](const auto& a, const auto& b) {
    const auto c = a.dot(b);
    const auto s = a.homogeneous().cross(b.homogeneous())(2);
    const auto dist = std::abs(std::atan2(s, c));
    return dist;
  };

  auto ds = DisjointSets(edges.size());
  auto visited = Image<std::uint8_t>{edges.sizes()};
  visited.flat_array().fill(0);

  auto statistics = std::vector<EdgeStatistics>(edges.size());

  // Collect the edgels and make as many sets as pixels.
  auto q = std::queue<Eigen::Vector2i>{};
  for (auto y = 0; y < edges.height(); ++y)
  {
    for (auto x = 0; x < edges.width(); ++x)
    {
      ds.make_set(index({x, y}));
      if (is_edgel({x, y}))
      {
        q.emplace(x, y);

        statistics[index({x, y})].total_mass = mag(x, y);

        statistics[index({x, y})].orientation_sum(0) += std::cos(ori(x, y));
        statistics[index({x, y})].orientation_sum(1) += std::sin(ori(x, y));

        statistics[index({x, y})].unnormalized_center(0) += x * mag(x, y);
        statistics[index({x, y})].unnormalized_center(1) += y * mag(x, y);

        statistics[index({x, y})].unnormalized_inertia(0) += x * x * mag(x, y);
        statistics[index({x, y})].unnormalized_inertia(1) += x * y * mag(x, y);
        statistics[index({x, y})].unnormalized_inertia(2) += y * y * mag(x, y);
      }
    }
  }

  // Neighborhood defined by 8-connectivity.
  const auto dir = std::array<Eigen::Vector2i, 8>{
      Eigen::Vector2i{1, 0},    //
      Eigen::Vector2i{1, 1},    //
      Eigen::Vector2i{0, 1},    //
      Eigen::Vector2i{-1, 1},   //
      Eigen::Vector2i{-1, 0},   //
      Eigen::Vector2i{-1, -1},  //
      Eigen::Vector2i{0, -1},   //
      Eigen::Vector2i{1, -1}    //
  };

  while (!q.empty())
  {
    const auto& p = q.front();
    visited(p) = 2;  // 2 = visited

    if (!is_edgel(p))
      throw std::runtime_error{"NOT AN EDGEL!"};

    // Find its corresponding node in the disjoint set.
    const auto node_p = ds.node(index(p));

    // Add nonvisited weak edges.
    for (const auto& d : dir)
    {
      const Eigen::Vector2i n = p + d;
      // Boundary conditions.
      if (n.x() < 0 || n.x() >= edges.width() ||  //
          n.y() < 0 || n.y() >= edges.height())
        continue;

      // Make sure that the neighbor is an edgel.
      if (!is_edgel(n))
        continue;

      const auto& comp_p = ds.component(index(n));
      const auto& comp_n = ds.component(index(n));
      const auto& up = vec(statistics[comp_p].angle());
      const auto& un = vec(statistics[comp_n].angle());

      // Merge component of p and component of n if angularly consistent.
      if (angular_distance(up, un) < angular_threshold)
      {
        const auto node_n = ds.node(index(n));
        ds.join(node_p, node_n);

        // Update the component statistics.
        const auto& comp_pn = ds.component(index(n));
        if (comp_pn == comp_p)
        {
          statistics[comp_pn].total_mass += statistics[comp_n].total_mass;

          statistics[comp_pn].orientation_sum +=
              statistics[comp_n].orientation_sum;
          statistics[comp_pn].unnormalized_center +=
              statistics[comp_n].unnormalized_center;
          statistics[comp_pn].unnormalized_inertia +=
              statistics[comp_n].unnormalized_inertia;
        }
        else  // if (comp_pn == comp_n)
        {
          statistics[comp_pn].total_mass += statistics[comp_n].total_mass;
          statistics[comp_pn].orientation_sum +=
              statistics[comp_p].orientation_sum;
          statistics[comp_pn].unnormalized_center +=
              statistics[comp_p].unnormalized_center;
          statistics[comp_pn].unnormalized_inertia +=
              statistics[comp_p].unnormalized_inertia;
        }
      }

      // Enqueue the neighbor n if it is not already enqueued
      if (visited(n) == 0)
      {
        // Enqueue the neighbor.
        q.emplace(n);
        visited(n) = 1;  // 1 = enqueued
      }
    }

    q.pop();
  }

  auto contours = std::map<int, std::vector<Point2i>>{};
  for (auto y = 0; y < edges.height(); ++y)
  {
    for (auto x = 0; x < edges.width(); ++x)
    {
      const auto p = Eigen::Vector2i{x, y};
      const auto index_p = index(p);
      if (is_edgel(p))
        contours[static_cast<int>(ds.component(index_p))].push_back(p);
    }
  }

  return contours;
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
      //"/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
#else
  // const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
  const auto video_filepath = "/home/david/Desktop/Datasets/ha/text_5.avi"s;
  // const auto video_filepath = "/home/david/Desktop/Datasets/ha/GH010175_converted.mp4"s;
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  const auto downscale_factor = 2;
  auto frame_gray32f = Image<float>{};

  // Show the local extrema.
  create_window(frame.sizes());
  set_antialiasing();

  constexpr auto high_threshold_ratio = 20e-2f;
  constexpr auto low_threshold_ratio = 10e-2f;
  constexpr auto angular_threshold = 20. / 180.f * M_PI;
  const auto sigma = std::sqrt(std::pow(1.6f, 2) - 1);
  const Eigen::Vector2i& p1 = frame.sizes() / 4;
  const Eigen::Vector2i& p2 = frame.sizes() * 3 / 4;

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
    toc("Hysteresis then Edge Grouping");
#endif


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


    // Display the quasi-straight edges.
    tic();
    // const auto labeled_edges = to_map(edges, edgels.sizes());
    const auto edge_colors = random_colors(edges);

    auto detection = frame;
    for (const auto& [label, points] : edges)
    {
      if (points.size() < 4)
        continue;

      for (const auto& p: points) {
        const Eigen::Vector2i q = p1 + downscale_factor * p;
        fill_rect(detection,                   //
                  q.x() - 1, q.y() - 1, 2, 2,  //
                  edge_colors.at(label));      //
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
