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

#include <algorithm>
#include <cmath>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Differential.hpp>
#include <drafts/Halide/LocalExtrema.hpp>
#include <drafts/Halide/Pyramids.hpp>
#include <drafts/Halide/RefineExtrema.hpp>
#include <drafts/Halide/Utilities.hpp>

#include <drafts/Halide/DominantGradientOrientations.hpp>
#include <drafts/Halide/Draw.hpp>
#include <drafts/Halide/Resize.hpp>
#include <drafts/Halide/SIFT.hpp>


namespace shakti = DO::Shakti;
namespace halide = DO::Shakti::HalideBackend;


namespace DO::Shakti::HalideBackend {

}  // namespace DO::Shakti::HalideBackend


auto test_on_image()
{
  const auto image_filepath =
#ifdef __APPLE__
      "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#else
      "/home/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#endif
  // "/Users/david/GitLab/DO-CV/sara/cpp/drafts/MatchPropagation/cpp/examples/shelves/shelf-1.jpg";
  auto image = sara::imread<float>(image_filepath);

  auto image_buffer = halide::as_buffer(image);
  auto out_buffer = Halide::Runtime::Buffer(;

  timer.restart();
  sift_extractor(image);
  SARA_DEBUG << "Halide SIFT computation time: "  //
             << timer.elapsed_ms() << " ms" << std::endl;
  SARA_CHECK(sift_extractor.pipeline.num_keypoints());

  // Show the local extrema.
  sara::create_window(image.sizes());
  sara::set_antialiasing();
  sara::display(image);
  draw_extrema(sift_extractor.pipeline.oriented_extrema);
  sara::get_key();
}

auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath = "/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
  // const auto video_filepath = "/home/david/Desktop/Datasets/ha/barberX.mp4"s;
#endif

  // Input and output from Sara.
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray32f = sara::Image<float>{frame.sizes()};

  const auto scale_factor = 1;
  auto frame_downsampled = sara::Image<float>{frame.sizes() / scale_factor};

  // Halide buffers.
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame);
  auto buffer_gray32f = halide::as_runtime_buffer<float>(frame_gray32f);

  auto sift_extractor = halide::SIFTExtractor{};

  // Show the local extrema.
  sara::create_window(frame_downsampled.sizes());
  sara::set_antialiasing();

  auto frames_read = 0;
  auto skip = 0;

  while (true)
  {
    sara::tic();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    sara::toc("Video Decoding");

    ++frames_read;
    if (frames_read % (skip + 1) != 0)
      continue;

    // Use parallelization and vectorization.
    sara::tic();
    shakti_halide_rgb_to_gray(buffer_rgb, buffer_gray32f);
    sara::toc("Grayscale");

    // Use parallelization and vectorization.
    sara::tic();
    halide::scale(frame_gray32f, frame_downsampled);
    sara::toc("Downsample");

    sara::tic();
// #define ORIGINAL
#ifdef ORIGINAL
    const auto [features, descriptors] =
        sara::compute_sift_keypoints(frame_downsampled);
#else
    sift_extractor(frame_downsampled);
#endif
    sara::toc("Oriented DoG");

    sara::tic();
    sara::display(frame_downsampled);
#ifdef ORIGINAL
    for (size_t i = 0; i != features.size(); ++i)
    {
      const auto color =
          features[i].extremum_type == sara::OERegion::ExtremumType::Max
              ? sara::Red8
              : sara::Blue8;
      features[i].draw(color);
    }
#else
    draw_extrema(sift_extractor.pipeline.oriented_extrema);
#endif
    sara::toc("Display");
  }
}


GRAPHICS_MAIN()
{
  test_on_image();
  // test_on_video();
  return 0;
}
