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

#include <unordered_map>
#include <unordered_set>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>

#include "Chessboard/ChessboardDetector.hpp"
#include "Utilities/ImageOrVideoReader.hpp"


namespace sara = DO::Sara;


auto draw_corner(sara::ImageView<sara::Rgb8>& display, const Corner<float>& c,
                 const int radius, const float scale, const sara::Rgb8& color,
                 int thickness) -> void
{
  const Eigen::Vector2i p1 = (scale * c.coords).array().round().cast<int>();
  sara::fill_circle(display, p1.x(), p1.y(), 1, sara::Yellow8);
  sara::draw_circle(display, p1.x(), p1.y(),
                    static_cast<int>(std::round(radius * scale)), color,
                    thickness);
};

auto draw_chessboard(sara::ImageView<sara::Rgb8>& display, const Chessboard& cb,
                     const std::vector<Corner<float>>& corners,
                     const std::vector<Square>& squares, const float scale,
                     const sara::Rgb8& color,
                     const int chessboard_edge_thickness) -> void
{
  for (const auto& row : cb)
  {
    for (const auto& sq : row)
    {
      if (sq.id == -1)
        continue;
      draw_square(corners, scale, display, squares[sq.id].v, sara::White8,
                  chessboard_edge_thickness + 1);
      draw_square(corners, scale, display, squares[sq.id].v, color,
                  chessboard_edge_thickness);
    }
  }
};

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

    const auto chessboard_edge_thickness = argc < 9 ? 2 : std::stoi(argv[8]);


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

#ifdef SHOW_ALL_CORNERS
      const auto& radius = detect._params.corner_filtering_radius;
      const auto num_corners = static_cast<int>(detect._corners.size());
#  pragma omp parallel for
      for (auto c = 0; c < num_corners; ++c)
      {
        const auto good = sara::is_seed_corner(   //
            detect._edges_adjacent_to_corner[c],  //
            detect._gradient_peaks_refined[c],    //
            detect._zero_crossings[c],            //
            detect.N);

        // Remove noisy corners to understand better the behaviour of the
        // algorithm.
        if (detect._edges_adjacent_to_corner[c].empty())
          continue;

        const auto& corner = detect._corners[c];
        draw_corner(display, corner, radius, scale,
                    good ? sara::Red8 : sara::Cyan8, 2);
      }
#endif

      const auto& corners = detect._corners;
      const auto& squares = detect._squares;
      const auto& scale = detect._params.downscale_factor;
      const auto& chessboards = detect._chessboards;
      const auto num_chessboards = static_cast<int>(chessboards.size());
#pragma omp parallel for
      for (auto c = 0; c < num_chessboards; ++c)
      {
        const auto color = sara::Rgb8(rand() % 255, rand() % 255, rand() % 255);
        const auto& cb = chessboards[c];
        draw_chessboard(display, cb, corners, squares, scale, color,
                        chessboard_edge_thickness);
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
