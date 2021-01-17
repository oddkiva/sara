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

#include <set>

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>


namespace sara = DO::Sara;


int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  // Input video.
  const auto video_filepath =
      argc < 2 ? "/Users/david/Desktop/Datasets/videos/sample10.mp4" : argv[1];
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();

  // Harris cornerness parameters.
  const auto sigma_D = std::sqrt(std::pow(1.6f, 2) - 1);
  const auto sigma_I = 3.f;
  const auto kappa = 10.f;

  sara::create_window(video_stream.sizes());
  sara::set_antialiasing();

  auto frames_read = 0;
  const auto skip = 0;
  while (true)
  {
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    ++frames_read;
    if (frames_read % (skip + 1) != 0)
      continue;

    const auto frame_gray = sara::downscale(frame.convert<float>(), 4);
    auto cornerness = sara::scale_adapted_harris_cornerness(  //
        frame_gray,                                           //
        sigma_I, sigma_D,                                     //
        kappa                                                 //
    );

    const auto cmin = cornerness.flat_array().minCoeff();
    const auto cmax = cornerness.flat_array().maxCoeff();
    cornerness.flat_array() = (cornerness.flat_array() - cmin) / (cmax - cmin);
    display(upscale(cornerness, 4));
  }

  return 0;
}


int main(int argc, char** argv)
{
  sara::GraphicsApplication app{argc, argv};
  app.register_user_main(__main);
  return app.exec();
}
