// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>


namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
#ifdef _WIN32
  const auto in_video_filepath = "C:/Users/David/Desktop/GOPR0542.mp4";
  const auto out_video_filepath = "C:/Users/David/Desktop/test.mkv";
#elif __APPLE__
  const auto in_video_filepath = "/Users/david/Desktop/Datasets/videos/sample10.mp4";
  const auto out_video_filepath = "/Users/david/Desktop/test.mkv";
#else
  const auto in_video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4";
  const auto out_video_filepath = "/home/david/Desktop/test.mkv";
#endif

  sara::VideoStream video_stream{in_video_filepath};
  sara::VideoWriter video_writer{out_video_filepath, video_stream.sizes(), 30};

  sara::create_window(video_stream.sizes());

  while (video_stream.read())
  {
    sara::display(video_stream.frame());

    sara::tic();
    video_writer.write(video_stream.frame());
    sara::toc("write");
  }

  return 0;
}
