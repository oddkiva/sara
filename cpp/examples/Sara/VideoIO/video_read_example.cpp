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

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <filesystem>


namespace fs = std::filesystem;
namespace sara = DO::Sara;


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

auto sara_graphics_main(int const argc, char** const argv) -> int
{
  using namespace std::string_literals;

  auto video_filepath = fs::path{
#ifdef _WIN32
      "C:/Users/David/Desktop/GOPR0542.MP4"
#elif __APPLE__
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"
#else
      "/home/david/Desktop/Datasets/sfm/Family.mp4"
#endif
  };

  if (argc >= 2)
    video_filepath = argv[1];

  auto video_stream = sara::VideoStream{video_filepath.string()};
  const auto video_frame = video_stream.frame();

  SARA_DEBUG << "Frame rate = " << video_stream.frame_rate() << std::endl;
  SARA_DEBUG << "Frame sizes = " << video_stream.sizes().transpose()
             << std::endl;
  SARA_DEBUG << "Frame rotation angle = " << video_stream.rotation_angle()
             << std::endl;

  sara::create_window(video_stream.sizes());
  while (true)
  {
    sara::tic();
    auto has_frame = video_stream.read();
    sara::toc("Read frame");

    if (!has_frame)
      break;

    sara::tic();
    sara::display(video_frame);
    sara::toc("Display");
  }

  sara::close_window();

  return 0;
}
