// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>


namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
  using namespace std::string_literals;

  // const std::string video_filepath = src_path("orion_1.mpg");
  // const auto video_filepath =
  //     "/home/david/Downloads/big-buck-bunny_trailer.webm"s;
  using namespace std::string_literals;
#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath =
      "/Users/david/Desktop/Datasets/humanising-autonomy/turn_bikes.mp4"s;
#else
  // const auto video_filepath = "/home/david/Desktop/test.mp4"s;
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
  //const auto video_filepath = "/home/david/Desktop/GOPR0542.MP4"s;
#endif

  sara::VideoStream video_stream{video_filepath};

  SARA_DEBUG << "Frame rate = " << video_stream.frame_rate() << std::endl;
  SARA_DEBUG << "Frame sizes = " << video_stream.sizes().transpose() << std::endl;

  while (video_stream.read())
  {
    if (sara::active_window() == nullptr)
      sara::set_active_window(sara::create_window(video_stream.sizes()));

    sara::display(video_stream.frame());
  }

  sara::close_window();

  return 0;
}
