// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO/VideoStream.hpp>

#include <DO/Shakti/ImageProcessing.hpp>
#include <DO/Shakti/MultiArray.hpp>
#include <DO/Shakti/Utilities/DeviceInfo.hpp>
#include <DO/Shakti/Utilities/Timer.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;


GRAPHICS_MAIN()
{
  auto devices = shakti::get_devices();
  devices.front().make_current_device();
  std::cout << devices.front() << std::endl;

  constexpr auto use_low_resolution_video = false;
  constexpr auto video_filepath =
      use_low_resolution_video
          ?
          // Video sample with image sizes (320 x 240).
          src_path("Segmentation/orion_1.mpg")
          :
          // Video samples with image sizes (1920 x 1080).
#ifdef _WIN32
          "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"
#else
          "/home/david/Desktop/test.mp4"
#endif
      ;
  std::cout << video_filepath << std::endl;

  sara::VideoStream video_stream{video_filepath};
  auto video_frame_index = int{0};
  auto video_frame = video_stream.frame();

  auto in_frame = sara::Image<float>{video_stream.sizes()};
  auto out_frame = sara::Image<float>{video_stream.sizes()};
  out_frame.flat_array().fill(0);

  // The pipeline is far from being optimized but despite that we almost get
  // real-time image processing.
  auto apply_gaussian_filter = shakti::GaussianFilter{3.f};

  auto cpu_timer = sara::Timer{};
  auto cpu_time = 0.;

  sara::create_window(video_frame.sizes());

  while (true)
  {
    cpu_timer.restart();
    if (!video_stream.read())
      break;
    cpu_time = cpu_timer.elapsed_ms();
    std::cout << "[CPU video decoding time] " << cpu_time << "ms" << std::endl;

    cpu_timer.restart();
    std::transform(video_frame.begin(), video_frame.end(), in_frame.begin(),
                   [](const sara::Rgb8& c) -> float {
                     auto gray = float{};
                     sara::smart_convert_color(c, gray);
                     return gray;
                   });
    cpu_time = cpu_timer.elapsed_ms();
    std::cout << "[CPU color conversion time] " << cpu_time << "ms"
              << std::endl;

    shakti::tic();
    apply_gaussian_filter(out_frame.data(), in_frame.data(),
                          in_frame.sizes().data());
    shakti::toc("GPU gaussian filter");

    sara::display(out_frame);

    ++video_frame_index;
    std::cout << std::endl;
  }

  return 0;
}
