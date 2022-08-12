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

#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>

#include <drafts/ChessboardDetection/ChessboardDetectorV2.hpp>

#include "Utilities/ImageOrVideoReader.hpp"


namespace sara = DO::Sara;


auto draw_corner(sara::ImageView<sara::Rgb8>& display,
                 const sara::Corner<float>& c,
                 const std::optional<std::vector<float>>& dominant_orientations,
                 const float downscale_factor, const float radius_factor,
                 const sara::Rgb8& color, int thickness) -> void
{
  const Eigen::Vector2i p1 =
      (downscale_factor * c.coords).array().round().cast<int>();
  const auto radius =
      static_cast<float>(M_SQRT2) * c.scale * downscale_factor * radius_factor;
  sara::draw_circle(display, p1.x(), p1.y(),
                    static_cast<int>(std::round(radius)), color, thickness);
  sara::fill_circle(display, p1.x(), p1.y(), 1, sara::Yellow8);

  if (!dominant_orientations.has_value())
    return;
  for (const auto& val : *dominant_orientations)
  {
    const Eigen::Vector2i p2 =
        (downscale_factor * c.coords + (radius * 2) * sara::dir(val))
            .array()
            .round()
            .cast<int>();
    sara::draw_arrow(display, p1.x(), p1.y(), p2.x(), p2.y(), color, thickness);
  }
}

auto draw_chessboard(sara::ImageView<sara::Rgb8>& display,  //
                     const sara::Chessboard& cb,
                     const std::vector<sara::Corner<float>>& corners,
                     const std::vector<sara::Square>& squares,
                     const float scale, const sara::Rgb8& color,
                     const int thickness) -> void
{
  for (const auto& row : cb)
  {
    for (const auto& sq : row)
    {
      if (sq.id == -1)
        continue;
      const auto& vertices = squares[sq.id].v;
      draw_square(display, vertices, corners, scale, sara::White8,
                  thickness + 1);
      draw_square(display, vertices, corners, scale, color, thickness);
    }
  }
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
    const auto num_scales = argc < 4 ? 2 : std::stoi(argv[3]);
    const auto scale_aa = argc < 5 ? 1.5f : std::stof(argv[4]);

    // Visual inspection option
    const auto show_edge_map = argc < 6  //
                                   ? false
                                   : static_cast<bool>(std::stoi(argv[5]));
    const auto pause = argc < 7  //
                           ? false
                           : static_cast<bool>(std::stoi(argv[6]));


    auto timer = sara::Timer{};
    auto video_stream = sara::ImageOrVideoReader{video_file};
    auto video_frame = video_stream.frame();
    auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};
    auto frame_number = -1;

    // Preprocessed image.
    auto frame_gray = sara::Image<float>{video_frame.sizes()};


    auto detect = sara::ChessboardDetectorV2{};
    detect.initialize_multiscale_harris_corner_detection_params(upscale,
                                                                num_scales);
    detect.initialize_filter_radius_according_to_scale();
    if (scale_aa < detect.gaussian_pyramid_params.scale_initial())
    {
      std::cerr << "Choose scale_aa > "
                << detect.gaussian_pyramid_params.scale_initial() << std::endl;
      return 1;
    }
    detect.scale_aa = scale_aa;
    detect.initialize_edge_detector();


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

        detect(frame_gray);
      }
      const auto elapsed_ms = timer.elapsed_ms();
      SARA_DEBUG << "Pipeline processing time = " << elapsed_ms << "ms"
                 << std::endl;

      auto display = sara::Image<sara::Rgb8>{};
      if (show_edge_map)
      {
        const auto edge_map = detect._edge_map.convert<float>();
        auto edge_map_us = sara::Image<float>{video_frame.sizes()};
        sara::resize_v2(edge_map, edge_map_us);

        display = edge_map_us.convert<sara::Rgb8>();

        const auto& endpoint_map = detect._endpoint_map;
        const auto& scale_aa = detect.scale_aa;

        const auto w = endpoint_map.width();
        const auto h = endpoint_map.height();
        const auto wh = w * h;
#pragma omp parallel for
        for (auto xy = 0; xy < wh; ++xy)
        {
          const auto y = xy / w;
          const auto x = xy - y * w;
          if (endpoint_map(x, y) != -1)
            sara::fill_circle(display, x * scale_aa, y * scale_aa, 2,
                              sara::Green8);
        }
      }
      else
        display = frame_gray.convert<sara::Rgb8>();

      static constexpr auto scale_image = 1.f;
      const auto& corners = detect._corners;
      const auto num_corners = static_cast<int>(corners.size());

#pragma omp parallel for
      for (auto c = 0; c < num_corners; ++c)
      {
        const auto color =
            detect._best_corners.find(c) != detect._best_corners.end()
                ? sara::Magenta8
                : sara::Cyan8;
        draw_corner(display,     //
                    corners[c],  //
                    std::nullopt, // detect._gradient_peaks_refined[c],
                    scale_image,           //
                    detect.radius_factor,  //
                    color, 2);
      }

      const auto& chessboards = detect._chessboards;
      const auto num_chessboards = static_cast<int>(chessboards.size());

      const auto& squares = detect._squares;
#  pragma omp parallel for
      for (auto c = 0; c < num_chessboards; ++c)
      {
        const auto color =
            c == 0 ? sara::Red8
                   : sara::Rgb8(rand() % 255, rand() % 255, rand() % 255);
        const auto& cb = chessboards[c];
        draw_chessboard(display, cb, corners, squares, 1.f, color, 5);
      }

      sara::draw_text(display, 80, 80, "Frame: " + std::to_string(frame_number),
                      sara::White8, 30, 0, false, true);
      sara::draw_text(display, 80, 120,
                      "Chessboards: " + std::to_string(chessboards.size()),
                      sara::White8, 30, 0, false, true);

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
