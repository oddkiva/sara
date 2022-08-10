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

#include <DO/Sara/Graphics.hpp>

#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/Harris.hpp>
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

auto create_image_pyramid(const Eigen::Vector2i& image_sizes,
                          const float start_scale,  //
                          const int num_scales)
    -> std::vector<sara::Image<float>>
{
  auto image_pyramid = std::vector<sara::Image<float>>(num_scales);

  auto scale_factor = 1 / start_scale;
  for (auto s = 0; s < num_scales; ++s)
  {
    const Eigen::Vector2i sizes =
        (image_sizes.cast<float>() * scale_factor).cast<int>();
    image_pyramid[s].resize(sizes);
    scale_factor *= 0.5f;
  }

  return image_pyramid;
}

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

    const auto upscale =
        argc < 3 ? false : static_cast<bool>(std::stoi(argv[2]));
    SARA_CHECK(upscale);
    const auto num_scales = argc < 4 ? 2 : std::stoi(argv[3]);
    SARA_CHECK(num_scales);

    // Visual inspection option
    const auto pause = argc < 5 ? false : static_cast<bool>(std::stoi(argv[4]));


    auto timer = sara::Timer{};
    auto video_stream = sara::ImageOrVideoReader{video_file};
    auto video_frame = video_stream.frame();
    auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};
    auto frame_number = -1;

    const auto create_image_pyramid_shortcut = [&video_frame, upscale,
                                                num_scales]() {
      return create_image_pyramid(video_frame.sizes(), upscale ? 0.5f : 1.f,
                                  num_scales);
    };

    const auto image_pyramid_params = sara::ImagePyramidParams(  //
        upscale ? -1 : 0,      // first octave index
        2,                     // 2 scales per octave
        2.f,                   // scale geom factor
        1,                     // image border
        upscale ? 0.5f : 1.f,  // scale camera
        scale_initial);

    // Preprocessed image.
    auto frame_gray = sara::Image<float>{video_frame.sizes()};


    // The feature pyramids.
#ifdef OLD
    auto frame_pyramid = create_image_pyramid_shortcut();
#else
    auto frame_pyramid = sara::ImagePyramid<float>{};
#endif
    auto grad_x_pyramid = create_image_pyramid_shortcut();
    auto grad_y_pyramid = create_image_pyramid_shortcut();
    auto cornerness_pyramid = create_image_pyramid_shortcut();

    auto corners_per_scale = std::vector<std::vector<sara::Corner<float>>>(
        cornerness_pyramid.size());

    auto fused_corners = std::vector<sara::Corner<float>>{};
    auto profiles = std::vector<Eigen::ArrayXf>{};
    auto zero_crossings = std::vector<std::vector<float>>{};

    struct CornerRef
    {
      std::int32_t id;
      float score;
      inline auto operator<(const CornerRef& other) const -> bool
      {
        return score < other.score;
      }
    };
    auto corners_adjacent_to_endpoints = std::vector<std::set<CornerRef>>{};

    auto profile_extractor = sara::CircularProfileExtractor{};

    auto ed = sara::EdgeDetector{};
    using sara::operator""_deg;
    ed.parameters.angular_threshold = static_cast<float>((10._deg).value);
    ed.parameters.high_threshold_ratio = 0.04f;
    ed.parameters.low_threshold_ratio = 0.02f;

    auto edge_map = sara::Image<float>{};
    auto endpoint_map = sara::Image<std::int32_t>{};
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

#ifdef OLD
        sara::tic();
        sara::apply_gaussian_filter(frame_gray, frame_pyramid[0], sigma_D);
        const auto sigma = std::sqrt(sara::square(2 * scale_initial) -
                                     sara::square(scale_initial));
        auto frame_before_downscale =
            frame_pyramid[0].compute<sara::Gaussian>(sigma);
        sara::scale(frame_before_downscale, frame_pyramid[1]);
        sara::toc("Frame pyramid");
#else
        sara::tic();
        frame_pyramid =
            sara::gaussian_pyramid(frame_gray, image_pyramid_params);
        sara::toc("Frame pyramid");
#endif

        sara::tic();
        for (auto o = 0; o < num_scales; ++o)
          sara::gradient(frame_pyramid(0, o),  //
                         grad_x_pyramid[o], grad_y_pyramid[o]);
        sara::toc("Gradient pyramid");

        sara::tic();
        for (auto o = 0; o < num_scales; ++o)
          cornerness_pyramid[o] = sara::harris_cornerness(  //
              grad_x_pyramid[o], grad_y_pyramid[o],         //
              sigma_I, kappa);
        sara::toc("Cornerness pyramid");

        for (auto o = 0; o < num_scales; ++o)
          corners_per_scale[o] = detect_corners(     //
              cornerness_pyramid[o],                 //
              grad_x_pyramid[o], grad_y_pyramid[o],  //
              scale_initial, sigma_I);


        sara::tic();
        static const auto scale_inter_delta = std::sqrt(  //
            sara::square(scale_inter) - sara::square(scale_initial));
        static const auto sizes_inter =
            (video_frame.sizes().cast<float>() / scale_inter)
                .array()
                .round()
                .cast<int>();
        auto frame_blurred_inter =
            frame_pyramid(0, 0).compute<sara::Gaussian>(scale_inter_delta);
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

        edge_map.resize(ed.pipeline.gradient_magnitude.sizes());
        edge_map.flat_array().fill(0);

        endpoint_map.resize(ed.pipeline.gradient_magnitude.sizes());
        endpoint_map.flat_array().fill(-1);

        const auto num_edges = static_cast<int>(edges.size());
#pragma omp parallel for
        for (auto e = 0; e < num_edges; ++e)
        {
          if (!is_strong_edge[e])
            continue;

          const auto& edge = edges[e];
          const auto& edge_ordered = sara::reorder_and_extract_longest_curve(edge);
          auto curve = std::vector<Eigen::Vector2d>(edge_ordered.size());
          std::transform(
              edge_ordered.begin(), edge_ordered.end(), curve.begin(),
              [](const auto& p) { return p.template cast<double>(); });
          if (curve.size() < 2)
            continue;

          const Eigen::Vector2i a =
              curve.front().array().round().matrix().cast<int>();
          const Eigen::Vector2i b =
              curve.back().array().round().matrix().cast<int>();

          for (const auto& p : edge)
            edge_map(p) = 1.;

          endpoint_map(a) = 2 * e;
          endpoint_map(b) = 2 * e + 1;
        }
        sara::toc("Edge filtering");

        sara::tic();
        fused_corners.clear();
        for (auto o = 0; o < num_scales; ++o)
        {
          const auto scale_factor = frame_pyramid.octave_scaling_factor(o);
          std::transform(corners_per_scale[o].begin(),
                         corners_per_scale[o].end(),
                         std::back_inserter(fused_corners),
                         [scale_factor](const sara::Corner<float>& corner) {
                           auto c = corner;
                           c.coords *= scale_factor;
                           c.score /= scale_factor;
                           c.scale *= scale_factor;
                           return c;
                         });
        }
        sara::scale_aware_nms(fused_corners, video_frame.sizes(),
                              radius_factor);
        sara::toc("Corner grouping and NMS");

#if 0
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
                const auto r = static_cast<int>(
                    std::round(M_SQRT2 * 2 * c.scale / scale_inter));

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
        sara::toc("Corner filtering (edge-vicinity)");
#endif

        sara::tic();
        {
          corners_adjacent_to_endpoints.clear();
          corners_adjacent_to_endpoints.resize(2 * edges.size());
          const auto num_corners = static_cast<int>(fused_corners.size());
          for (auto c = 0; c < num_corners; ++c)
          {
            const auto& corner = fused_corners[c];

            const Eigen::Vector2i p = (corner.coords / scale_inter)
                                          .array()
                                          .round()
                                          .matrix()
                                          .cast<int>();
            const auto r = static_cast<int>(std::round(
                M_SQRT2 * radius_factor * corner.scale / scale_inter));

            const auto umin = std::clamp(p.x() - r, 0, edge_map.width());
            const auto umax = std::clamp(p.x() + r, 0, edge_map.width());
            const auto vmin = std::clamp(p.y() - r, 0, edge_map.height());
            const auto vmax = std::clamp(p.y() + r, 0, edge_map.height());
            for (auto v = vmin; v <= vmax; ++v)
            {
              for (auto u = umin; u <= umax; ++u)
              {
                const auto endpoint_id = endpoint_map(u, v);
                if (endpoint_id != -1)
                  corners_adjacent_to_endpoints[endpoint_id].insert(
                      CornerRef{c, corner.score});
              }
            }
          }
        }
        sara::toc("Topological linking");

        sara::tic();
        {
          auto best_corner_ids = std::unordered_set<int>{};
          for (const auto& corners_adj_to_endpoint: corners_adjacent_to_endpoints)
          {
            if (corners_adj_to_endpoint.empty())
              continue;
            const auto best_corner = corners_adj_to_endpoint.rbegin();
            best_corner_ids.insert(best_corner->id);
          }
          auto fused_corners_filtered = std::vector<sara::Corner<float>>{};
          std::transform(best_corner_ids.begin(), best_corner_ids.end(),
                         std::back_inserter(fused_corners_filtered),
                         [&fused_corners](const auto& id) {
                           return fused_corners[id];
                         });
          fused_corners_filtered.swap(fused_corners);
        }
        sara::toc("Corner filtering (end-point)");


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
          // TODO: check this... this was guessed without verification.
          const auto offset = static_cast<int>(
              std::pow(2, num_scales - static_cast<int>(upscale)));
          const auto w = frame_gray.width();
          const auto h = frame_gray.height();
          if (!(r + offset <= p.x() && p.x() < w - r - offset &&  //
                r + offset <= p.y() && p.y() < h - r - offset))
            continue;

          const auto radius = M_SQRT2 * corner.scale * radius_factor;
          const auto& frame =
              upscale ? frame_pyramid(0, 1) : frame_pyramid(0, 0);
          profiles[c] = profile_extractor(
              frame, fused_corners[c].coords.cast<double>(), radius);

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


#define VIEW_EDGE_MAP
#ifdef VIEW_EDGE_MAP

      auto edge_map_us = sara::Image<float>{video_frame.sizes()};
      sara::resize_v2(edge_map, edge_map_us);

      auto display = edge_map_us.convert<sara::Rgb8>();
      for (auto y = 0; y < endpoint_map.height(); ++y)
        for (auto x = 0; x < endpoint_map.width(); ++x)
          if (endpoint_map(x, y) != -1)
            sara::fill_circle(display, x * scale_inter, y * scale_inter, 2,
                              sara::Green8);
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
