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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/VideoIO.hpp>


namespace sara = DO::Sara;


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

  auto f = std::vector{
      sara::Image<float>{video_frame.sizes()},
      sara::Image<float>{video_frame.sizes() / 2},  //
  };

  auto f_conv = std::vector{
      sara::Image<float>{video_frame.sizes()},
      sara::Image<float>{video_frame.sizes() / 2},  //
  };

  auto segmentation_map = sara::Image<std::uint8_t>{video_frame.sizes()};

  static constexpr auto tolerance_parameter = 0.0f;

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

    sara::tic();
    sara::from_rgb8_to_gray32f(video_frame, f[0]);
 // #define DOWNSAMPLE
#ifdef DOWNSAMPLE
    sara::scale(f[0], f[1]);
#endif
    sara::toc("Grayscale conversion");

    sara::tic();
#ifdef DOWNSAMPLE
    sara::apply_gaussian_filter(f[1], f_conv[1], 16.f);
    sara::enlarge(f_conv[1], f_conv[0]);
    std::transform(std::execution::par_unseq, f_conv[0].begin(),
                   f_conv[0].end(), f[0].begin(), segmentation_map.begin(),
                   [](const auto& mean, const auto& val) {
                     return val > (mean - tolerance_parameter) ? 255 : 0;
                   });
#else
    sara::gaussian_adaptive_threshold(f[0], 64.f, 2.f, tolerance_parameter, segmentation_map);
#endif
    sara::toc("Adaptive thresholding");

    sara::display(segmentation_map);
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
