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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "libavcodec/avcodec.h"
#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/mathematics.h"


namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
  using namespace std;
  using namespace DO::Sara;

  const string video_filepath = src_path("orion_1.mpg");

  VideoStream video_stream(video_filepath);
  auto video_frame = Image<Rgb8>{video_stream.sizes()};

  while (true)
  {
    video_stream >> video_frame;
    if (!active_window())
      create_window(video_frame.sizes());

    if (!video_frame.data())
      break;
    display(video_frame);
  }
}


// GRAPHICS_MAIN()
// {
//   using namespace std::string_literals;
// 
//   const auto video_filepath =
//       "/home/david/Desktop/humanising-autonomy/barberX.mp4"s;
// 
//   sara::VideoStream2 video_stream;
//   video_stream.open(video_filepath);
// 
//   while (video_stream.read())
//   {
//     const auto frame = video_stream.frame();
// 
//     if (sara::active_window() == nullptr)
//       sara::create_window(frame.sizes());
// 
//     sara::display(frame);
//   }
// 
//   sara::close_window();
// 
//   return 0;
// }
