// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <set>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/FeatureDetectors/LineSegmentDetector.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>

#include <omp.h>


using namespace std;
using namespace DO::Sara;


auto test_on_image()
{
  // Read an image.
  const auto image =
      imread<float>(src_path("../../../../data/sunflowerField.jpg"));

  const auto sigma = sqrt(square(1.6f) - square(0.5f));
  auto image_curr = deriche_blur(image, sigma);

  create_window(image.sizes());
  display(image_curr);

  // THE line segment detector.
  auto lsd = LineSegmentDetector{};
  lsd.parameters.high_threshold_ratio = 5e-2f;
  lsd.parameters.low_threshold_ratio = 2e-2f;
  lsd.parameters.angular_threshold = static_cast<float>(20. / 180. * M_PI);

  for (auto s = 0; s < 500; ++s)
  {
    // Detect line segments.
    lsd(image_curr);

    // Display the fitted lines.
    fill_rect(0, 0, image_curr.width(), image_curr.height(), Black8);
    const auto curve_colors = random_colors(lsd.pipeline.contours);
    for (auto i = 0u; i < lsd.pipeline.line_segments.size(); ++i)
    {
      const auto& [success, l] = lsd.pipeline.line_segments[i];
      if (success)
        draw_line(l.p1(), l.p2(),                              //
                  curve_colors.at(lsd.pipeline.curve_ids[i]),  //
                  /* line_width */ 2);
    }

    millisleep(1);

    // Blur.
    const auto delta = std::pow(2.f, 1 / 100.f);
    const auto sigma = 1.6f * sqrt(pow(delta, static_cast<float>(2 * s + 2)) -
                                   pow(delta, static_cast<float>(2 * s)));
    image_curr = deriche_blur(image_curr, sigma);
  }

  get_key();
}

auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  // const auto video_filepath =
  // "/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
  const auto video_filepath =
      //"/Users/david/Desktop/Datasets/videos/sample1.mp4"s;
      //"/Users/david/Desktop/Datasets/videos/sample4.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  const auto downscale_factor = 1;
  auto f = Image<float>{frame.sizes()};
  auto f_ds = Image<float>{frame.sizes() / downscale_factor};

  // Show the local extrema.
  create_window(frame.sizes() / downscale_factor);
  set_antialiasing();

  // THE line segment detector.
  auto lsd = LineSegmentDetector{};
  lsd.parameters.high_threshold_ratio = 20e-2f;
  lsd.parameters.low_threshold_ratio = 10e-2f;
  lsd.parameters.angular_threshold = static_cast<float>(20. / 180. * M_PI);
  lsd.parameters.polish_line_segments = false;

  // Loop over the video.
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

    // Preprocess the image.
    {
      // Convert to grayscale
      from_rgb8_to_gray32f(frame, f);

      // Blur.
      f = f.compute<Gaussian>(std::sqrt(square(1.6f) - 1));

      // Downscale.
      if (downscale_factor > 1)
      {
        scale(f, f_ds);
        f_ds.swap(f);
      }
    }

    // Detect the line segments.
    lsd(f);

    // Display the fitted lines.
    auto disp = f.convert<Rgb8>();
    const auto num_lines = static_cast<int>(lsd.pipeline.line_segments.size());
#pragma omp parallel for
    for (auto i = 0; i < num_lines; ++i)
    {
      const auto& [success, l] = lsd.pipeline.line_segments[i];
      if (!success || l.length() < 20)
        continue;
      draw_line(disp, l.p1().cast<float>(), l.p2().cast<float>(),  //
                Rgb8(rand() % 255, rand() % 255, rand() % 255),    //
                /* line_width */ 2);
    }
    display(disp);
  }
}


GRAPHICS_MAIN()
{
  omp_set_num_threads(omp_get_max_threads());

  // test_on_image();
  test_on_video();
  return 0;
}
