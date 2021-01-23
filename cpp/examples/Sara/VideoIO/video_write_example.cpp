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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>


namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
  const auto in_video_filepath = "/Users/david/Desktop/Datasets/videos/sample2.mp4";
  sara::VideoStream video_stream{in_video_filepath};

  const auto out_video_filepath = "/Users/david/Desktop/test.mp4";
  sara::VideoWriter video_writer{out_video_filepath, video_stream.sizes(), 30};

  while (video_stream.read())
  {
    sara::display(video_stream.frame());
    video_writer.write(video_stream.frame());
  }

  return 0;
}
