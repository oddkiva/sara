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
#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Halide/Utilities.hpp>

#include "shakti_gaussian_convolution_v2.h"
#include "shakti_halide_rgb_to_gray.h"
#include "shakti_polar_gradient_2d_32f_v2.h"
#include "shakti_scale_32f.h"

#include <boost/filesystem.hpp>

#include <omp.h>


using namespace std;
using namespace DO::Sara;


namespace v2 {

  struct EdgeDetector
  {
    //! @brief intermediate data.
    struct Pipeline
    {
      Image<float> gradient_magnitude;
      Image<float> gradient_orientation;

      Image<std::uint8_t> edge_map;
      std::map<int, std::vector<Point2i>> edges;

      std::vector<std::vector<Eigen::Vector2i>> edges_as_list;
      std::vector<std::vector<Eigen::Vector2d>> edges_simplified;

    } pipeline;

    struct Parameters
    {
      //! @brief Canny edge parameters.
      //! @{
      float high_threshold_ratio = 5e-2f;
      float low_threshold_ratio = 2e-2f;
      //! @}

      //! @brief Angle tolerance for connected edgel grouping.
      float angular_threshold = static_cast<float>(20. / 180.f * M_PI);

      //! @brief Edge simplification parameters.
      bool simplify_edges = true;
      double eps = 1.;
      double collapse_threshold = 2e-2;
      bool collapse_adaptive = true;
    } parameters;

    EdgeDetector() = default;

    EdgeDetector(const EdgeDetector::Parameters& params)
      : parameters{params}
    {
    }

    auto operator()(Halide::Runtime::Buffer<float>& image_buffer_4d) -> void
    {
      auto& mag = pipeline.gradient_magnitude;
      auto& ori = pipeline.gradient_orientation;
      if (mag.width() != image_buffer_4d.width() || mag.height() != image_buffer_4d.height())
        mag.resize(image_buffer_4d.width(), image_buffer_4d.height());
      if (ori.width() != image_buffer_4d.width() || ori.height() != image_buffer_4d.height())
        ori.resize(image_buffer_4d.width(), image_buffer_4d.height());

      // Sara tensors.
      auto mag_tensor_view = tensor_view(mag).reshape(
          Eigen::Vector4i{1, 1, mag.height(), mag.width()});
      auto ori_tensor_view = tensor_view(ori).reshape(
          Eigen::Vector4i{1, 1, ori.height(), ori.width()});

      // Halide buffers.
      auto mag_buffer_4d =
          DO::Shakti::HalideBackend::as_runtime_buffer(mag_tensor_view);
      auto ori_buffer_4d =
          DO::Shakti::HalideBackend::as_runtime_buffer(ori_tensor_view);

      tic();
      // image_buffer_4d.set_host_dirty();
      shakti_polar_gradient_2d_32f_v2(image_buffer_4d, mag_buffer_4d, ori_buffer_4d);
      mag_buffer_4d.copy_to_host();
      ori_buffer_4d.copy_to_host();
      toc("Polar Coordinates");

      tic();
      const auto& grad_mag = pipeline.gradient_magnitude;
      const auto& grad_mag_max = grad_mag.flat_array().maxCoeff();
      const auto& high_thres = grad_mag_max * parameters.high_threshold_ratio;
      const auto& low_thres = grad_mag_max * parameters.low_threshold_ratio;
      pipeline.edge_map = suppress_non_maximum_edgels(  //
          pipeline.gradient_magnitude,                  //
          pipeline.gradient_orientation,                //
          high_thres, low_thres);
      toc("Thresholding");

      tic();
      pipeline.edges = perform_hysteresis_and_grouping(  //
          pipeline.edge_map,                             //
          pipeline.gradient_orientation,                 //
          parameters.angular_threshold);
      toc("Hysteresis & Edgel Grouping");

      tic();
      const auto& edges = pipeline.edges;
      auto& edges_as_list = pipeline.edges_as_list;
      edges_as_list.resize(edges.size());
      std::transform(edges.begin(), edges.end(), edges_as_list.begin(),
                     [](const auto& e) { return e.second; });
      toc("To vector");

      if (parameters.simplify_edges)
      {
        tic();
        auto& edges_simplified = pipeline.edges_simplified;
        edges_simplified.resize(edges_as_list.size());
#pragma omp parallel for
        for (auto i = 0u; i < edges_as_list.size(); ++i)
        {
          const auto& edge =
              reorder_and_extract_longest_curve(edges_as_list[i]);

          auto edges_converted = std::vector<Eigen::Vector2d>(edge.size());
          std::transform(
              edge.begin(), edge.end(), edges_converted.begin(),
              [](const auto& p) { return p.template cast<double>(); });

          edges_simplified[i] =
              ramer_douglas_peucker(edges_converted, parameters.eps);
        }
        toc("Longest Curve Extraction & Simplification");

        // tic();
        // #pragma omp parallel for
        // for (auto i = 0u; i < edges_simplified.size(); ++i)
        //   if (edges_simplified[i].size() > 2)
        //     edges_simplified[i] = collapse(edges_simplified[i],
        //                                    grad_mag,
        //                                    parameters.collapse_threshold,
        //                                    parameters.collapse_adaptive);
        // toc("Vertex Collapse");
        //
        // tic();
        // auto& edges_refined = edges_simplified;
        // #pragma omp parallel for
        // for (auto i = 0u; i < edges_refined.size(); ++i)
        //   for (auto& p : edges_refined[i])
        //     p = refine(grad_mag, p.cast<int>()).cast<double>();
        // toc("Refine Edge Localisation");
      }
    }
  };

}  // namespace v2


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}


int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  // Parse command line.
  const auto video_filepath = argc >= 2
                                  ? argv[1]
#ifdef _WIN32
                                  : "C:/Users/David/Desktop/GOPR0542.MP4"s;
#elif __APPLE__
                                  : "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
                                  : "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif
  const auto downscale_factor = argc >= 3 ? std::atoi(argv[2]) : 2;

  // OpenMP.
  omp_set_num_threads(omp_get_max_threads());

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray32f = Image<float>{frame.sizes()};
  auto frame_gray32f_convolved = Image<float>{frame.sizes()};
  auto frame_gray32f_downscaled =
      Image<float>{frame.sizes() / downscale_factor};

  auto frame_buffer =
      DO::Shakti::HalideBackend::as_interleaved_runtime_buffer(frame);
  auto frame_gray32f_buffer =
      DO::Shakti::HalideBackend::as_runtime_buffer(frame_gray32f);

  auto frame_gray32f_tensor_view =
      tensor_view(frame_gray32f)
          .reshape(Eigen::Vector4i{1, 1, frame_gray32f.height(),
                                   frame_gray32f.width()});
  auto frame_gray32f_convolved_tensor_view =
      tensor_view(frame_gray32f_convolved)
          .reshape(Eigen::Vector4i{1, 1, frame_gray32f_convolved.height(),
                                   frame_gray32f_convolved.width()});
  auto frame_gray32f_downscaled_tensor_view =
      tensor_view(frame_gray32f_downscaled)
          .reshape(Eigen::Vector4i{1, 1, frame_gray32f_downscaled.height(),
                                   frame_gray32f_downscaled.width()});

  auto frame_gray32f_buffer_4d =
      DO::Shakti::HalideBackend::as_runtime_buffer(frame_gray32f_tensor_view);
  auto frame_gray32f_convolved_buffer_4d =
      DO::Shakti::HalideBackend::as_runtime_buffer(
          frame_gray32f_convolved_tensor_view);
  auto frame_gray32f_downscaled_buffer_4d =
      DO::Shakti::HalideBackend::as_runtime_buffer(
          frame_gray32f_downscaled_tensor_view);

  // Output save.
  namespace fs = boost::filesystem;
  const auto basename = fs::basename(video_filepath);
  VideoWriter video_writer{
#ifdef __APPLE__
      "/Users/david/Desktop/" + basename + ".edge-detection.mp4",
#else
      "/home/david/Desktop/" + basename + ".edge-detection.mp4",
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

  auto ed = v2::EdgeDetector{{
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
    SARA_DEBUG << "Processing frame " << frames_read << std::endl;

    tic();
    shakti_halide_rgb_to_gray(frame_buffer, frame_gray32f_buffer);
    toc("Grayscale");

    tic();
    frame_gray32f_buffer_4d.set_host_dirty();
    shakti_gaussian_convolution_v2(frame_gray32f_buffer_4d, sigma, 4,
                                   frame_gray32f_convolved_buffer_4d);
    toc("Blur");

    if (downscale_factor > 1)
    {
      tic();
      shakti_scale_32f(frame_gray32f_convolved_buffer_4d,
                       frame_gray32f_downscaled_buffer_4d.width(),
                       frame_gray32f_downscaled_buffer_4d.height(),
                       frame_gray32f_downscaled_buffer_4d);
      toc("Downscale");
    }

    if (downscale_factor > 1)
      ed(frame_gray32f_downscaled_buffer_4d);
    else
      ed(frame_gray32f_convolved_buffer_4d);
    const auto& edges_simplified = ed.pipeline.edges_simplified;

    tic();
    for (const auto& e : edges_simplified)
      if (e.size() >= 2 && length(e) > 5)
        draw_polyline(frame, e, Red8, Eigen::Vector2d(0, 0),
                      float(downscale_factor));
    display(frame);
    toc("Display");

    tic();
    video_writer.write(frame);
    toc("Video Write");
  }

  return 0;
}
