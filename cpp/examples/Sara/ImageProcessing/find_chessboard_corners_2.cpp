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

#include <execution>
#include <unordered_map>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/VideoIO.hpp>


namespace sara = DO::Sara;


auto mean_colors(const std::map<int, std::vector<Eigen::Vector2i>>& regions,
                 const sara::ImageView<sara::Rgb8>& image)
{
  auto colors = std::unordered_map<int, sara::Rgb8>{};
  for (const auto& [label, points] : regions)
  {
    const auto num_points = static_cast<float>(points.size());
    Eigen::Vector3f color = Eigen::Vector3f::Zero();
    for (const auto& p : points)
      color += image(p).cast<float>();
    color /= num_points;

    colors[label] = color.cast<std::uint8_t>();
  }
  return colors;
}


auto __main(int argc, char** argv) -> int
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

  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto frame_number = -1;

  auto f = sara::Image<float>{video_frame.sizes()};
  auto f_conv = sara::Image<float>{video_frame.sizes()};

#ifdef ADAPTIVE_THRESHOLDING
  auto segmentation_map = sara::Image<std::uint8_t>{video_frame.sizes()};
  static constexpr auto tolerance_parameter = 0.0f;
#else
  static const auto color_threshold = std::sqrt(sara::square(2.f) * 3);
#endif


  while (video_stream.read())
  {
    ++frame_number;
    if (frame_number % 3 != 0)
      continue;

    if (sara::active_window() == nullptr)
    {
      sara::create_window(video_frame.sizes());
      sara::set_antialiasing();
    }
    SARA_CHECK(frame_number);

#ifdef ADAPTIVE_THRESHOLDING
    sara::tic();
    sara::from_rgb8_to_gray32f(video_frame, f);
    sara::toc("Grayscale conversion");

    sara::tic();
    sara::gaussian_adaptive_threshold(f, 64.f, 2.f, tolerance_parameter,
                                      segmentation_map);
    sara::toc("Adaptive thresholding");

    sara::display(segmentation_map);
#else
    // Watershed.
    sara::tic();
    const auto regions = sara::color_watershed(video_frame, color_threshold);
    sara::toc("Watershed");

    // Display the good regions.
    const auto colors = mean_colors(regions, video_frame);
    auto partitioning = sara::Image<sara::Rgb8>{video_frame.sizes()};
    for (const auto& [label, points] : regions)
    {
      // Show big segments only.
      for (const auto& p : points)
        partitioning(p) = points.size() < 100 ? sara::Black8 : colors.at(label);
    }
    sara::display(partitioning);
#endif

    sara::draw_text(80, 80, std::to_string(frame_number), sara::White8, 60, 0,
                    false, true);
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
