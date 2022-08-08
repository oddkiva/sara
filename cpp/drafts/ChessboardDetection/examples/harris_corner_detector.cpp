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


auto draw_corner(sara::ImageView<sara::Rgb8>& display,
                 const sara::Corner<float>& c, const int radius,
                 const float scale, const sara::Rgb8& color, int thickness)
    -> void
{
  const Eigen::Vector2i p1 = (scale * c.coords).array().round().cast<int>();
  sara::draw_circle(display, p1.x(), p1.y(),
                    static_cast<int>(std::round(radius * scale)), color,
                    thickness);
  sara::fill_circle(display, p1.x(), p1.y(), 1, sara::Yellow8);
};


auto detect_corners(const sara::ImageView<float>& cornerness,
                    const sara::ImageView<float>& grad_x,
                    const sara::ImageView<float>& grad_y)
{
  static constexpr auto cornerness_adaptive_thres = 1e-5f;
  static constexpr auto corner_filtering_radius = 7;

  const auto corners_quantized = select(  //
      cornerness,                         //
      cornerness_adaptive_thres,          //
      corner_filtering_radius);
  sara::toc("Corner selection");

  sara::tic();
  auto corners = std::vector<sara::Corner<float>>{};
  std::transform(
      corners_quantized.begin(), corners_quantized.end(),
      std::back_inserter(corners),
      [&grad_x, &grad_y](const sara::Corner<int>& c) -> sara::Corner<float> {
        const auto p = sara::refine_junction_location_unsafe(
            grad_x, grad_y, c.coords, corner_filtering_radius);
        return {p, c.score};
      });
  sara::nms(corners, cornerness.sizes(), corner_filtering_radius);
  sara::toc("Corner refinement");

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

    static constexpr auto sigma_D = 0.8f;
    static constexpr auto sigma_I = 0.8f;
    static constexpr auto kappa = 0.04f;

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
    auto profiles = std::array<std::vector<Eigen::ArrayXf>, 2>{};
    auto zero_crossings = std::array<std::vector<std::vector<float>>, 2>{};

    auto fused_corners = std::vector<sara::Corner<float>>{};

    auto profile_extractor = sara::CircularProfileExtractor{};
    profile_extractor.circle_radius = sigma_I * 3;

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
        const auto sigma =
            std::sqrt(sara::square(2 * sigma_D) - sara::square(sigma_D));
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

        sara::tic();
        for (auto i = 0; i < 2; ++i)
          corners[i] = detect_corners(cornerness_pyramid[i],  //
                                      grad_x_pyramid[i], grad_y_pyramid[i]);

        sara::toc("Corners detection");

        sara::tic();
        for (auto i = 0; i < 2; ++i)
        {
          profiles[i].clear();
          zero_crossings[i].clear();
          profiles[i].resize(corners[i].size());
          zero_crossings[i].resize(corners[i].size());
          auto num_corners = static_cast<int>(corners[i].size());
#pragma omp parallel for
          for (auto c = 0; c < num_corners; ++c)
          {
            const auto& p = corners[i][c].coords;
            const auto& r = profile_extractor.circle_radius;
            const auto w = frame_pyramid[i].width();
            const auto h = frame_pyramid[i].height();
            if (!(r + 1 <= p.x() && p.x() < w - r - 1 &&  //
                  r + 1 <= p.y() && p.y() < h - r - 1))
              continue;
            profiles[i][c] = profile_extractor(  //
                frame_pyramid[i],                //
                corners[i][c].coords.cast<double>());
            zero_crossings[i][c] = sara::localize_zero_crossings(
                profiles[i][c], profile_extractor.num_circle_sample_points);
          }
        }
        sara::toc("Circular intensity profile");

        sara::tic();
        {
          for (auto i = 0; i < 2; ++i)
          {
            SARA_CHECK(i);
            auto corners_filtered = std::vector<sara::Corner<float>>{};
            auto profiles_filtered = std::vector<Eigen::ArrayXf>{};
            auto zero_crossings_filtered = std::vector<std::vector<float>>{};

            for (auto c = 0u; c < corners[i].size(); ++c)
            {
              if (sara::is_good_x_corner(zero_crossings[i][c]))
              {
                corners_filtered.emplace_back(corners[i][c]);
                profiles_filtered.emplace_back(profiles[i][c]);
                zero_crossings_filtered.emplace_back(zero_crossings[i][c]);
              }
            }

            corners_filtered.swap(corners[i]);
            profiles_filtered.swap(profiles[i]);
            zero_crossings_filtered.swap(zero_crossings[i]);
          }
        }
        sara::toc("Corner filtering from intensity profile");

        fused_corners.clear();
        std::copy(corners[0].begin(), corners[0].end(),
                  std::back_inserter(fused_corners));
        std::transform(corners[1].begin(), corners[1].end(),
                       std::back_inserter(fused_corners),
                       [](const sara::Corner<float>& corner) {
                         auto c = corner;
                         c.coords *= 2;
                         c.score /= 2;
                         return c;
                       });
        sara::nms(fused_corners, video_frame.sizes(), 5 * sigma_I);
      }
      const auto pipeline_time = timer.elapsed_ms();
      SARA_DEBUG << "Processing time = " << pipeline_time << "ms" << std::endl;

#if 0
      for (auto i = 1; i >= 0; --i)
      {
        const auto scale = std::pow(2, i);
        const auto& corners_i = corners[i];
        const auto num_corners = static_cast<int>(corners_i.size());
#pragma omp parallel for
        for (auto c = 0; c < num_corners; ++c)
          draw_corner(video_frame, corners_i[c], sigma_I * scale * 2, scale,
                      sara::Red8, 2);
      }
#else
        const auto num_corners = static_cast<int>(fused_corners.size());
#  pragma omp parallel for
        for (auto c = 0; c < num_corners; ++c)
          draw_corner(video_frame, fused_corners[c], sigma_I * 5, 1.f,
                      sara::Red8, 2);
#endif
      sara::display(video_frame);
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
