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
#include <DO/Sara/Geometry/Algorithms/Polyline.hpp>
#include <DO/Sara/Geometry/Tools/Utilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <future>

#include <omp.h>


using namespace std;
using namespace DO::Sara;


constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}

constexpr long double operator"" _deg(long double x)
{
  return x * M_PI / 180;
}


// ========================================================================== //
// Shape Statistics of the Edge.
// ========================================================================== //
struct Rectangle
{
  const Eigen::Vector2d& center;
  const Eigen::Matrix2d& axes;
  const Eigen::Vector2d& lengths;

  Rectangle(const Eigen::Vector2d& c,  //
            const Eigen::Matrix2d& a,  //
            const Eigen::Vector2d& l)
    : center{c}
    , axes{a}
    , lengths{l}
  {
  }

  auto length_ratio() const
  {
    return lengths(0) / lengths(1);
  }

  auto line() const
  {
    return Projective::line(center.homogeneous().eval(),
                            (center + axes.col(0)).homogeneous().eval());
  }

  auto draw(ImageView<Rgb8>& detection, const Rgb8& color,  //
            const Point2d& c1, const double s) const -> void
  {
    const Vector2d u = axes.col(0);
    const Vector2d v = axes.col(1);
    auto p = std::array<Vector2d, 4>{
        c1 + s * (center + (lengths(0) + 0) * u + (lengths(1) + 0) * v),
        c1 + s * (center - (lengths(0) + 0) * u + (lengths(1) + 0) * v),
        c1 + s * (center - (lengths(0) + 0) * u - (lengths(1) + 0) * v),
        c1 + s * (center + (lengths(0) + 0) * u - (lengths(1) + 0) * v),
    };

    draw_line(detection, p[0].x(), p[0].y(), p[1].x(), p[1].y(), color, 2);
    draw_line(detection, p[1].x(), p[1].y(), p[2].x(), p[2].y(), color, 2);
    draw_line(detection, p[2].x(), p[2].y(), p[3].x(), p[3].y(), color, 2);
    draw_line(detection, p[3].x(), p[3].y(), p[0].x(), p[0].y(), color, 2);
  }
};


// ========================================================================== //
// Edge Grouping By Alignment
// ========================================================================== //
using Edge = std::vector<Eigen::Vector2d>;

struct EdgeAttributes
{
  const std::vector<Edge>& edges;
  const std::vector<Eigen::Vector2d>& centers;
  const std::vector<Eigen::Matrix2d>& axes;
  const std::vector<Eigen::Vector2d>& lengths;
};

struct EndPointGraph
{
  const EdgeAttributes& edge_attrs;

  // List of end points.
  std::vector<Point2d> endpoints;
  // Edge IDs to which the end point belongs to.
  std::vector<std::size_t> edge_ids;

  // Connect the end point to another point.
  // Cannot be in the same edge ids.
  Eigen::MatrixXd score;


  EndPointGraph(const EdgeAttributes& attrs)
    : edge_attrs{attrs}
  {
    endpoints.reserve(2 * edge_attrs.edges.size());
    edge_ids.reserve(2 * edge_attrs.edges.size());
    for (auto i = 0u; i < edge_attrs.edges.size(); ++i)
    {
      const auto& e = edge_attrs.edges[i];
      if (e.size() < 2)
        continue;

      if (length(e) < 5)
        continue;

      const auto& theta = std::abs(std::atan2(edge_attrs.axes[i](1, 0),  //
                                              edge_attrs.axes[i](0, 0)));
      if (theta < 5._deg || std::abs(M_PI - theta) < 5._deg)
        continue;

      endpoints.emplace_back(e.front());
      endpoints.emplace_back(e.back());

      edge_ids.emplace_back(i);
      edge_ids.emplace_back(i);
    }

    score = Eigen::MatrixXd(endpoints.size(), endpoints.size());
    score.fill(std::numeric_limits<double>::infinity());
  }

  auto edge(std::size_t i) const -> const Edge&
  {
    const auto& edge_id = edge_ids[i];
    return edge_attrs.edges[edge_id];
  }

  auto rect(std::size_t i) const -> Rectangle
  {
    const auto& edge_id = edge_ids[i];
    return Rectangle{
        edge_attrs.centers[edge_id],  //
        edge_attrs.axes[edge_id],     //
        edge_attrs.lengths[edge_id]   //
    };
  }

  auto mark_plausible_alignments() -> void
  {
    // Tolerance of X degrees in the alignment error.
    const auto thres = std::cos(20._deg);

    for (auto i = 0u; i < endpoints.size() / 2; ++i)
    {
      for (auto k = 0; k < 2; ++k)
      {
        const auto& ik = 2 * i + k;
        const auto& p_ik = endpoints[ik];

        const auto& r_ik = rect(ik);
        const auto& c_ik = r_ik.center;

        for (auto j = i + 1; j < endpoints.size() / 2; ++j)
        {
          // The closest and most collinear point.
          for (int l = 0; l < 2; ++l)
          {
            const auto& jl = 2 * j + l;
            const auto& p_jl = endpoints[jl];

            const auto& r_jl = rect(jl);
            const auto& c_jl = r_jl.center;

            // Particular case:
            if ((p_ik - p_jl).squaredNorm() < 1e-3)
            {
              // Check the plausibility that the end points are mutually
              // aligned?
              const auto dir = std::array<Eigen::Vector2d, 2>{
                  (p_ik - c_ik).normalized(),  //
                  (c_jl - p_jl).normalized()   //
              };

              const auto cosine = dir[0].dot(dir[1]);
              if (cosine < thres)
                continue;

              score(ik, jl) = 0;
            }
            else
            {
              const auto dir = std::array<Eigen::Vector2d, 3>{
                  (p_ik - c_ik).normalized(),  //
                  (p_jl - p_ik).normalized(),  //
                  (c_jl - p_jl).normalized()   //
              };
              const auto cosines = std::array<double, 2>{
                  dir[0].dot(dir[1]),
                  dir[1].dot(dir[2]),
              };

              if (cosines[0] + cosines[1] < 2 * thres)
                continue;

              if (Projective::point_to_line_distance(p_ik.homogeneous().eval(),
                                                     r_jl.line()) > 10 &&
                  Projective::point_to_line_distance(p_jl.homogeneous().eval(),
                                                     r_ik.line()) > 10)
                continue;

              // We need this to be as small as possible.
              const auto dist = (p_ik - p_jl).norm();

              // We really need to avoid accidental connections like these
              // situations. Too small edges and too far away, there is little
              // chance it would correspond to a plausible alignment.
              if (length(edge(ik)) + length(edge(jl)) < 20 && dist > 20)
                continue;

              if (dist > 50)
                continue;

              score(ik, jl) = dist;
            }
          }
        }
      }
    }
  }

  auto group() const
  {
    const auto n = score.rows();

    auto ds = DisjointSets(n);

    for (auto i = 0; i < n; ++i)
      ds.make_set(i);

    for (auto i = 0; i < n; ++i)
      for (auto j = i; j < n; ++j)
        if (score(i, j) != std::numeric_limits<double>::infinity())
          ds.join(ds.node(i), ds.node(j));

    auto groups = std::map<std::size_t, std::vector<std::size_t>>{};
    for (auto i = 0; i < n; ++i)
    {
      const auto c = ds.component(i);
      groups[c].push_back(i);
    }

    return groups;
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

  const auto video_filepath =
      argc == 2
          ? argv[1]
#ifdef _WIN32
          : "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
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

  // Show the local extrema.
  create_window(frame.sizes());
  set_antialiasing();

  constexpr float high_threshold_ratio = 20._percent;
  constexpr float low_threshold_ratio = high_threshold_ratio / 2.;
  constexpr float angular_threshold = 20. / 180. * M_PI;
  const auto sigma = std::sqrt(std::pow(1.2f, 2) - 1);
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
    auto& edges_refined = ed.pipeline.edges_simplified;

    tic();
    // TODO: split only if the inertias matrix is becoming isotropic.
    edges_refined = split(edges_refined, 10. * M_PI / 180.);
    toc("Edge Split");


    tic();
    // TODO: figure out why the linear directional mean is shaky.
    // auto ldms = std::vector<double>(edges_refined.size());
    // The rectangle approximation.
    auto centers = std::vector<Vector2d>(edges_refined.size());
    auto inertias = std::vector<Matrix2d>(edges_refined.size());
    auto axes = std::vector<Matrix2d>(edges_refined.size());
    auto lengths = std::vector<Vector2d>(edges_refined.size());
#pragma omp parallel for
    for (auto i = 0u; i < edges_refined.size(); ++i)
    {
      const auto& e = edges_refined[i];
      if (e.size() < 2)
        continue;

      // ldms[i] = linear_directional_mean(e);
      centers[i] = center_of_mass(e);
      inertias[i] = matrix_of_inertia(e, centers[i]);
      const auto svd = inertias[i].jacobiSvd(Eigen::ComputeFullU);
      axes[i] = svd.matrixU();
      lengths[i] = svd.singularValues().cwiseSqrt();
    }
    SARA_CHECK(edges_refined.size());
    toc("Edge Shape Statistics");

    tic();
    const auto edge_attributes = EdgeAttributes{.edges = edges_refined,
                                                .centers = centers,
                                                .axes = axes,
                                                .lengths = lengths};
    auto endpoint_graph = EndPointGraph{edge_attributes};
    endpoint_graph.mark_plausible_alignments();
    toc("Alignment Computation");

    const Eigen::Vector2d p1d = p1.cast<double>();
    const auto& s = downscale_factor;

    auto detection = Image<Rgb8>{frame};
    detection.flat_array().fill(Black8);

    for (const auto& e : edges_refined)
      if (e.size() >= 2)
        draw_polyline(detection, e, Blue8, p1d, s);

    // Draw alignment-based connections.
    auto remap = [&](const auto p) -> Vector2d { return p1d + s * p; };
    const auto& score = endpoint_graph.score;
    for (auto i = 0; i < score.rows(); ++i)
    {
      for (auto j = i + 1; j < score.cols(); ++j)
      {
        const auto& pi = endpoint_graph.endpoints[i];
        const auto& pj = endpoint_graph.endpoints[j];

        if (score(i, j) != std::numeric_limits<double>::infinity())
        {
          const auto pi1 = remap(pi);
          const auto pj1 = remap(pj);
          draw_line(detection, pi1.x(), pi1.y(), pj1.x(), pj1.y(), Yellow8, 2);
          draw_circle(detection, pi1.x(), pi1.y(), 3., Yellow8, 3);
          draw_circle(detection, pj1.x(), pj1.y(), 3., Yellow8, 3);
        }
      }
    }

    display(detection);
  }

  return 0;
}
