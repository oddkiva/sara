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

#include <drafts/ChessboardDetection/ChessboardDetector.hpp>

#include "Utilities/ImageOrVideoReader.hpp"

#include <optional>


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
  sara::fill_circle(display, p1.x(), p1.y(), thickness, sara::Red8);

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
    const auto pause = argc < 5  //
                           ? false
                           : static_cast<bool>(std::stoi(argv[4]));


    auto timer = sara::Timer{};
    auto video_stream = sara::ImageOrVideoReader{video_file};
    auto video_frame = video_stream.frame();
    auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};
    auto frame_number = -1;

    // Preprocessed image.
    auto frame_gray = sara::Image<float>{video_frame.sizes()};


    // Set up the corner detector.
    auto detect = sara::ChessboardDetector{};
    detect.initialize_multiscale_harris_corner_detection_params(upscale,
                                                                num_scales);
    detect.initialize_filter_radius_according_to_scale();
    detect.edge_detection_params.high_threshold_ratio = 0.2f;
    detect.edge_detection_params.low_threshold_ratio = 0.1f;
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

        detect.calculate_feature_pyramids(frame_gray);
        detect.extract_corners();
        detect.detect_edges();
        detect.filter_edges();
        detect.group_and_filter_corners();
        detect.link_corners_to_edge_endpoints_topologically();
        detect.filter_corners_topologically();
        detect.calculate_circular_intensity_profiles();
        detect.calculate_orientation_histograms();
      }
      const auto elapsed_ms = timer.elapsed_ms();
      SARA_DEBUG << "Pipeline processing time = " << elapsed_ms << "ms"
                 << std::endl;

      auto& display = video_frame;

      static constexpr auto scale_image = 1.f;
      const auto& corners = detect._corners;
      const auto num_corners = static_cast<int>(corners.size());
      SARA_CHECK(num_corners);

#pragma omp parallel for
      for (auto c = 0; c < num_corners; ++c)
      {
        if (detect._gradient_peaks_refined[c].size() < 2u)
          continue;
        draw_corner(display,               //
                    corners[c],            //
                    std::nullopt,          // detect._gradient_peaks_refined[c],
                    scale_image,           //
                    detect.radius_factor,  //
                    sara::Red8, 4);
      }
      SARA_DEBUG << "OK" << std::endl;

      sara::draw_text(display, 80, 80, "Frame: " + std::to_string(frame_number),
                      sara::White8, 30, 0, false, true);

      sara::display(display);
      if (pause)
        sara::get_key();

      sara::imwrite(display, "/Users/david/Desktop/corners.png");
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
