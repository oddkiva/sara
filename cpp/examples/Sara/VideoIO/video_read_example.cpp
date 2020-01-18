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
  using namespace std::string_literals;

  const auto video_filepath =
      "/home/david/Desktop/humanising-autonomy/barberX.mp4"s;
  //const std::string video_filepath = src_path("orion_1.mpg");

  sara::VideoStream video_stream{video_filepath};

  SARA_DEBUG << "Frame rate = " << video_stream.frame_rate() << std::endl;
  SARA_DEBUG << "Frame sizes = " << video_stream.sizes().transpose() << std::endl;

  auto video_frame = sara::ImageView<sara::Rgb8>{};

  while (video_stream.read2())
  {
    if (sara::active_window() == nullptr)
      sara::create_window(video_stream.sizes());

    sara::display(video_stream.frame());
  }

  sara::close_window();

  return 0;
}
