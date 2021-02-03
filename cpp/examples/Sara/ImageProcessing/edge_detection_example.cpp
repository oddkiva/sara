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
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/ImageProcessing/EdgeGrouping.hpp>

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
struct OrientedBox
{
  const Eigen::Vector2d& center;
  const Eigen::Matrix2d& axes;
  const Eigen::Vector2d& lengths;

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
    const auto p = std::array<Vector2d, 4>{
        c1 + s * (center + (lengths(0) + 0) * u + (lengths(1) + 0) * v),
        c1 + s * (center - (lengths(0) + 0) * u + (lengths(1) + 0) * v),
        c1 + s * (center - (lengths(0) + 0) * u - (lengths(1) + 0) * v),
        c1 + s * (center + (lengths(0) + 0) * u - (lengths(1) + 0) * v),
    };
    auto pi = std::array<Vector2i, 4>{};
    std::transform(p.begin(), p.end(), pi.begin(),
                   [](const Vector2d& v) { return v.cast<int>(); });

    draw_line(detection, pi[0].x(), pi[0].y(), pi[1].x(), pi[1].y(), color, 2);
    draw_line(detection, pi[1].x(), pi[1].y(), pi[2].x(), pi[2].y(), color, 2);
    draw_line(detection, pi[2].x(), pi[2].y(), pi[3].x(), pi[3].y(), color, 2);
    draw_line(detection, pi[3].x(), pi[3].y(), pi[0].x(), pi[0].y(), color, 2);
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

  // Show the local extrema.
  create_window(frame.sizes());
  set_antialiasing();

  constexpr float high_threshold_ratio = static_cast<float>(20._percent);
  constexpr float low_threshold_ratio = static_cast<float>(high_threshold_ratio / 2.);
  constexpr float angular_threshold = static_cast<float>(20. / 180. * M_PI);
  const auto sigma = std::sqrt(std::pow(1.2f, 2) - 1);
// #define CROP
#ifdef CROP
  const Eigen::Vector2i& p1 = frame.sizes() / 4;
  const Eigen::Vector2i& p2 = frame.sizes() * 3 / 4;
#else
  const Eigen::Vector2i& p1 = Eigen::Vector2i::Zero();
  const Eigen::Vector2i& p2 = frame.sizes();
#endif

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
    for (auto i = 0; i < static_cast<int>(edges_refined.size()); ++i)
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

#ifdef PERFORM_EDGE_GROUPING
    tic();
    const auto edge_attributes = EdgeAttributes{
        edges_refined,  //
        centers,        //
        axes,           //
        lengths         //
    };
    auto endpoint_graph = EndPointGraph{edge_attributes};
    endpoint_graph.mark_plausible_alignments();
    toc("Alignment Computation");
#endif

    const Eigen::Vector2d p1d = p1.cast<double>();
    const auto s = static_cast<float>(downscale_factor);

    auto detection = Image<Rgb8>{frame};
    detection.flat_array().fill(Black8);

    for (const auto& e : edges_refined)
      if (e.size() >= 2)
        draw_polyline(detection, e, Blue8, p1d, s);

#ifdef PERFORM_EDGE_GROUPING
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
          const auto pi1 = remap(pi).cast<int>();
          const auto pj1 = remap(pj).cast<int>();
          draw_line(detection, pi1.x(), pi1.y(), pj1.x(), pj1.y(), Yellow8, 2);
          draw_circle(detection, pi1.x(), pi1.y(), 3, Yellow8, 3);
          draw_circle(detection, pj1.x(), pj1.y(), 3, Yellow8, 3);
        }
      }
    }
#endif

    display(detection);
  }

  return 0;
}
