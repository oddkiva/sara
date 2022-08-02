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

#include "Chessboard/ChessboardDetector.hpp"
#include "Utilities/ImageOrVideoReader.hpp"


namespace sara = DO::Sara;


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
    const auto check_edge_map = argc < 4
                                    ? false  //
                                    : static_cast<bool>(std::stoi(argv[3]));

    // Setup the detection parameters.
    auto params = sara::ChessboardDetector::Parameters{};
    if (argc >= 5)
      params.downscale_factor = std::stof(argv[4]);
    if (argc >= 6)
      params.cornerness_adaptive_thres = std::stof(argv[5]);
    if (argc >= 7)
    {
      const auto value = std::stoi(argv[6]);
      if (value != -1)
        params.corner_filtering_radius = value;
      else
        params.set_corner_nms_radius();
    }
    else
      params.set_corner_nms_radius();
    if (argc >= 8)
    {
      const auto value = std::stoi(argv[7]);
      if (value != -1)
        params.corner_edge_linking_radius = value;
      else
        params.set_corner_edge_linking_radius_to_corner_filtering_radius();
    }
    else
      params.set_corner_edge_linking_radius_to_corner_filtering_radius();


    auto timer = sara::Timer{};
    auto video_stream = sara::ImageOrVideoReader{video_file};
    auto video_frame = video_stream.frame();
    auto frame_gray = sara::Image<float>{video_frame.sizes()};
    auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};
    auto frame_number = -1;

    auto detect = sara::ChessboardDetector{params};

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
      const auto pipeline_time = timer.elapsed_ms();
      SARA_DEBUG << "Processing time = " << pipeline_time << "ms" << std::endl;

      sara::tic();
      if (check_edge_map)
      {
        // Resize
        auto display_32f_ds = detect._ed.pipeline.edge_map.convert<float>();
        auto display_32f = sara::Image<float>{video_frame.sizes()};
        sara::scale(display_32f_ds, display_32f);

        display = display_32f.convert<sara::Rgb8>();
      }
      else
        display = frame_gray.convert<sara::Rgb8>();

      const auto num_corners = static_cast<int>(detect._corners.size());
#pragma omp parallel for
      for (auto c = 0; c < num_corners; ++c)
      {
        const auto& p = detect._corners[c];
        const auto good = sara::is_seed_corner(   //
            detect._edges_adjacent_to_corner[c],  //
            detect._gradient_peaks_refined[c],    //
            detect._zero_crossings[c],            //
            detect.N);

        // Remove noisy corners to understand better the behaviour of the
        // algorithm.
        if (detect._edges_adjacent_to_corner[c].empty())
          continue;

        const auto& radius = detect._params.corner_filtering_radius;
        const auto& scale = detect._params.downscale_factor;
        sara::fill_circle(display,
                          static_cast<int>(std::round(scale * p.coords.x())),
                          static_cast<int>(std::round(scale * p.coords.y())), 1,
                          sara::Yellow8);
        sara::draw_circle(display,
                          static_cast<int>(std::round(scale * p.coords.x())),
                          static_cast<int>(std::round(scale * p.coords.y())),
                          static_cast<int>(std::round(radius * scale)),
                          good ? sara::Red8 : sara::Cyan8, 2);
      }
      sara::draw_text(display, 80, 80, std::to_string(frame_number),
                      sara::White8, 60, 0, false, true);

      const auto& corners = detect._corners;
      const auto& lines = detect._lines;
      const auto& scale = detect._params.downscale_factor;
      for (const auto& line : lines)
      {
        for (auto i = 0u; i < line.size() - 1; ++i)
        {
          const Eigen::Vector2f a = corners[line[i]].coords * scale;
          const Eigen::Vector2f b = corners[line[i + 1]].coords * scale;
          sara::draw_line(display, a, b, sara::Cyan8, 2);
        }
      }

      const auto draw_square = [&corners, scale,
                                &display](const auto& square,  //
                                          const auto& color,   //
                                          const int thickness) {
        for (auto i = 0; i < 4; ++i)
        {
          const Eigen::Vector2f a = corners[square[i]].coords * scale;
          const Eigen::Vector2f b = corners[square[(i + 1) % 4]].coords * scale;
          sara::draw_line(display, a, b, color, thickness);
        }
      };

      for (const auto& square : detect._white_squares)
        draw_square(square, sara::Red8, 3);

      for (const auto& square : detect._black_squares)
        draw_square(square, sara::Green8, 2);

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
