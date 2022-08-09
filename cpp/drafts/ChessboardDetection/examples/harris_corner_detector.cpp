// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <omp.h>

#include <DO/Sara/FeatureDetectors/Harris.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/Differential.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>

#include <drafts/ChessboardDetection/ChessboardDetector.hpp>
#include <drafts/ChessboardDetection/Corner.hpp>
#include <drafts/ChessboardDetection/NonMaximumSuppression.hpp>

#include "Utilities/ImageOrVideoReader.hpp"


namespace sara = DO::Sara;


// Harris corner parameters.
static constexpr auto scale_image = 1.f;
static constexpr auto scale_delta = 0.8f;
static const auto scale_initial = std::sqrt(sara::square(scale_image) +  //
                                            sara::square(scale_delta));
static const auto scale_inter = std::sqrt(2.f) * scale_initial;

static constexpr auto sigma_D = scale_delta;
static constexpr auto sigma_I = 0.8f;
static constexpr auto kappa = 0.04f;

// Circular profile extractor parameters.
static constexpr auto radius_factor = 3.f;


auto draw_corner(sara::ImageView<sara::Rgb8>& display,
                 const sara::Corner<float>& c, const float downscale_factor,
                 const sara::Rgb8& color, int thickness) -> void
{
  const Eigen::Vector2i p1 =
      (downscale_factor * c.coords).array().round().cast<int>();
  const auto radius =
      static_cast<float>(M_SQRT2) * c.scale * downscale_factor * radius_factor;
  sara::draw_circle(display, p1.x(), p1.y(),
                    static_cast<int>(std::round(radius)), color, thickness);
  sara::fill_circle(display, p1.x(), p1.y(), 1, sara::Yellow8);
};


auto detect_corners(const sara::ImageView<float>& cornerness,
                    const sara::ImageView<float>& grad_x,
                    const sara::ImageView<float>& grad_y,  //
                    const float image_scale,               //
                    const float sigma_I)
{
  static constexpr auto cornerness_adaptive_thres = 1e-5f;
  static constexpr auto corner_filtering_radius = 7;

  sara::tic();
  const auto corners_quantized = select(  //
      cornerness,                         //
      image_scale, sigma_I,               //
      cornerness_adaptive_thres,          //
      corner_filtering_radius);
  sara::toc("Corner selection");

  sara::tic();
  auto corners = std::vector<sara::Corner<float>>(corners_quantized.size());
  const auto num_corners = static_cast<int>(corners.size());
#pragma omp parallel for
  for (auto c = 0; c < num_corners; ++c)
  {
    const auto& cq = corners_quantized[c];
    const auto p = sara::refine_junction_location_unsafe(
        grad_x, grad_y, cq.coords, corner_filtering_radius);
    // TODO: interpolate the cornerness.
    corners[c] = {p, cq.score, image_scale * sigma_I};
  }
  sara::toc("Corner refinement");

  sara::tic();
  sara::scale_aware_nms(corners, cornerness.sizes(), radius_factor);
  sara::toc("Corner NMS");

  return corners;
}

auto __main(int argc, char** argv) -> int
{
  try
  {
    omp_set_num_threads(omp_get_max_threads());

#ifdef _WIN32
    const auto video_file = sara::select_video_file_from_dialog_box();
    if (video_file.empty())
      return 1;
#else
    if (argc < 2)
      return 1;
    const auto video_file = std::string{argv[1]};
#endif

    // Visual inspection option
    const auto pause = argc < 3 ? false : static_cast<bool>(std::stoi(argv[2]));


    auto timer = sara::Timer{};
    auto video_stream = sara::ImageOrVideoReader{video_file};
    auto video_frame = video_stream.frame();
    auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};
    auto frame_number = -1;

    auto frame_gray = sara::Image<float>{video_frame.sizes()};
    auto frame_pyramid = std::array{
        sara::Image<float>{video_frame.sizes()},
        sara::Image<float>{video_frame.sizes() / 2}  //
    };

    auto grad_x_pyramid = std::array{
        sara::Image<float>{video_frame.sizes()},
        sara::Image<float>{video_frame.sizes() / 2}  //
    };
    auto grad_y_pyramid = std::array{
        sara::Image<float>{video_frame.sizes()},
        sara::Image<float>{video_frame.sizes() / 2}  //
    };

    auto cornerness_pyramid = std::array{
        sara::Image<float>{video_frame.sizes()},
        sara::Image<float>{video_frame.sizes() / 2}  //
    };

    auto corners = std::array<std::vector<sara::Corner<float>>, 2>{};

    auto fused_corners = std::vector<sara::Corner<float>>{};
    auto profiles = std::vector<Eigen::ArrayXf>{};
    auto zero_crossings = std::vector<std::vector<float>>{};


    auto profile_extractor = sara::CircularProfileExtractor{};

    auto ed = sara::EdgeDetector{};
    using sara::operator""_deg;
    ed.parameters.angular_threshold = static_cast<float>((10._deg).value);
    ed.parameters.high_threshold_ratio = 0.04f;
    ed.parameters.low_threshold_ratio = 0.02f;

    auto edge_map = sara::Image<float>{};
    auto is_strong_edge = std::vector<std::uint8_t>{};

    while (video_stream.read())
    {
      ++frame_number;
      if (frame_number % 3 != 0)
        continue;
      SARA_DEBUG << "Frame #" << frame_number << std::endl;

      if (sara::active_window() == nullptr)
      {
        sara::create_window(video_frame.sizes(), video_file);
        sara::set_antialiasing();
      }

      timer.restart();
      {
        sara::tic();
        sara::from_rgb8_to_gray32f(video_frame, frame_gray);
        sara::toc("Grayscale conversion");

        sara::tic();
        sara::apply_gaussian_filter(frame_gray, frame_pyramid[0], sigma_D);
        const auto sigma = std::sqrt(sara::square(2 * scale_initial) -
                                     sara::square(scale_initial));
        auto frame_before_downscale =
            frame_pyramid[0].compute<sara::Gaussian>(sigma);
        sara::scale(frame_before_downscale, frame_pyramid[1]);
        sara::toc("Frame pyramid");

        sara::tic();
        for (auto i = 0; i < 2; ++i)
          sara::gradient(frame_pyramid[i], grad_x_pyramid[i],
                         grad_y_pyramid[i]);
        sara::toc("Gradient pyramid");

        sara::tic();
        for (auto i = 0; i < 2; ++i)
          cornerness_pyramid[i] = sara::harris_cornerness(  //
              grad_x_pyramid[i], grad_y_pyramid[i],         //
              sigma_I, kappa);
        sara::toc("Cornerness pyramid");

        for (auto i = 0; i < 2; ++i)
          corners[i] = detect_corners(cornerness_pyramid[i],  //
                                      grad_x_pyramid[i], grad_y_pyramid[i],
                                      scale_initial, sigma_I);


        sara::tic();
        static const auto scale_inter_delta =
            std::sqrt(sara::square(scale_inter) - sara::square(scale_initial));
        static const auto sizes_inter =
            (video_frame.sizes().cast<float>() / scale_inter)
                .array()
                .round()
                .cast<int>();
        auto frame_blurred_inter =
            frame_pyramid[0].compute<sara::Gaussian>(scale_inter_delta);
        auto frame_inter = sara::Image<float>{sizes_inter};
        sara::resize_v2(frame_blurred_inter, frame_inter);
        ed(frame_inter);
        sara::toc("Edge detection");

        sara::tic();
        is_strong_edge.clear();
        const auto& edges = ed.pipeline.edges_as_list;
        std::transform(
            edges.begin(), edges.end(), std::back_inserter(is_strong_edge),
            [&ed](const auto& e) {
              static constexpr auto strong_edge_thres = 4.f / 255.f;
              return sara::is_strong_edge(ed.pipeline.gradient_magnitude, e,
                                          strong_edge_thres);
            });

        edge_map = sara::Image<float>{
            ed.pipeline.gradient_magnitude.sizes()  //
        };
        edge_map.flat_array().fill(0);

        const auto num_edges = static_cast<int>(edges.size());
#pragma omp parallel for
        for (auto e = 0; e < num_edges; ++e)
        {
          if (!is_strong_edge[e])
            continue;

          const auto& edge = edges[e];
          for (const auto& p : edge)
            edge_map(p) = 1.;
        }
        sara::toc("Edge filtering");

        sara::tic();
        fused_corners.clear();
        std::copy(corners[0].begin(), corners[0].end(),
                  std::back_inserter(fused_corners));
        std::transform(corners[1].begin(), corners[1].end(),
                       std::back_inserter(fused_corners),
                       [](const sara::Corner<float>& corner) {
                         auto c = corner;
                         c.coords *= 2;
                         c.score /= 2;
                         c.scale *= 2;
                         return c;
                       });
        sara::scale_aware_nms(fused_corners, video_frame.sizes(),
                              radius_factor);
        sara::toc("Corner grouping and NMS");

        sara::tic();
        {
          auto fused_corners_filtered = std::vector<sara::Corner<float>>{};
          std::copy_if(
              fused_corners.begin(), fused_corners.end(),
              std::back_inserter(fused_corners_filtered),
              [&edge_map](const sara::Corner<float>& c) {
                const Eigen::Vector2i p = (c.coords / scale_inter)
                                              .array()
                                              .round()
                                              .matrix()
                                              .cast<int>();
                const auto r = static_cast<int>(std::round(
                    M_SQRT2 * 2 * c.scale / scale_inter));

                const auto umin = std::clamp(p.x() - r, 0, edge_map.width());
                const auto umax = std::clamp(p.x() + r, 0, edge_map.width());
                const auto vmin = std::clamp(p.y() - r, 0, edge_map.height());
                const auto vmax = std::clamp(p.y() + r, 0, edge_map.height());

                for (auto v = vmin; v < vmax; ++v)
                  for (auto u = umin; u < umax; ++u)
                    if (edge_map(u, v))
                      return true;
                return false;
              });
          fused_corners_filtered.swap(fused_corners);
        }
        sara::toc("Corner filtering on edge-vicinity-criterion");

        sara::tic();
        profiles.clear();
        zero_crossings.clear();
        profiles.resize(fused_corners.size());
        zero_crossings.resize(fused_corners.size());
        auto num_corners = static_cast<int>(fused_corners.size());
#pragma omp parallel for
        for (auto c = 0; c < num_corners; ++c)
        {
          const auto& corner = fused_corners[c];
          const auto& p = corner.coords;
          const auto& r = profile_extractor.circle_radius;
          const auto w = frame_pyramid[0].width();
          const auto h = frame_pyramid[0].height();
          if (!(r + 1 <= p.x() && p.x() < w - r - 1 &&  //
                r + 1 <= p.y() && p.y() < h - r - 1))
            continue;

          const auto radius = M_SQRT2 * corner.scale * radius_factor;
          profiles[c] = profile_extractor(             //
              frame_pyramid[0],                        //
              fused_corners[c].coords.cast<double>(),  //
              radius);

          zero_crossings[c] = sara::localize_zero_crossings(
              profiles[c], profile_extractor.num_circle_sample_points);
        }
        sara::toc("Circular intensity profile");

        sara::tic();
        {
          auto corners_filtered = std::vector<sara::Corner<float>>{};
          auto profiles_filtered = std::vector<Eigen::ArrayXf>{};
          auto zero_crossings_filtered = std::vector<std::vector<float>>{};

          for (auto c = 0u; c < fused_corners.size(); ++c)
          {
            if (sara::is_good_x_corner(zero_crossings[c]))
            {
              corners_filtered.emplace_back(fused_corners[c]);
              profiles_filtered.emplace_back(profiles[c]);
              zero_crossings_filtered.emplace_back(zero_crossings[c]);
            }
          }

          corners_filtered.swap(fused_corners);
          profiles_filtered.swap(profiles);
          zero_crossings_filtered.swap(zero_crossings);
        }
        sara::toc("Corner filtering from intensity profile");
      }
      const auto pipeline_time = timer.elapsed_ms();
      SARA_DEBUG << "Processing time = " << pipeline_time << "ms" << std::endl;


// #define VIEW_EDGE_MAP
#ifdef VIEW_EDGE_MAP

      auto edge_map_us = sara::Image<float>{video_frame.sizes()};
      sara::resize_v2(edge_map, edge_map_us);

      auto display = edge_map_us.convert<sara::Rgb8>();
#else
      auto display = frame_gray.convert<sara::Rgb8>();
#endif

      const auto num_corners = static_cast<int>(fused_corners.size());
#pragma omp parallel for
      for (auto c = 0; c < num_corners; ++c)
        draw_corner(display, fused_corners[c], scale_image, sara::Red8, 2);
      sara::display(display);
      if (pause)
        sara::get_key();
    }
  }
  catch (std::exception& e)
  {
    // Harris corner parameters.
    std::cout << e.what() << std::endl;
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
