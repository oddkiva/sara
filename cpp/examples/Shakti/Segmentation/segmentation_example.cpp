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

#include <functional>
#include <random>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Segmentation.hpp>
#include <DO/Shakti/Utilities/DeviceInfo.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;

using namespace std;
using namespace sara;


void draw_grid(const Vector2i& sizes, const Vector2i block_sizes)
{
  const auto& w = sizes.x();
  const auto& h = sizes.y();

  Vector2i steps =
      ((sizes.array() + block_sizes.array() - 1) / block_sizes.array())
          .matrix();

  cout << sizes.transpose() << endl;
  cout << block_sizes.transpose() << endl;
  cout << steps.transpose() << endl;

  for (auto x = 0; x < steps.x(); ++x)
    draw_line(Point2i{x * block_sizes.x(), 0}, Point2i{x * block_sizes.x(), h},
              Green8, 1);

  for (auto y = 0; y < steps.y(); ++y)
    draw_line(Point2i{0, y * block_sizes.y()}, Point2i{w, y * block_sizes.y()},
              Green8, 1);

  for (auto y = 0; y < steps.y(); ++y)
    for (auto x = 0; x < steps.x(); ++x)
    {
      fill_circle(Point2i{x * block_sizes.x() + block_sizes.x() / 2,
                          y * block_sizes.y() + block_sizes.y() / 2},
                  3, Green8);
    }
}


void demo_on_image()
{
  auto image_path = src_path("Segmentation/sunflower_field.jpg");
  auto image = sara::imread<Rgba32f>(image_path);

  // Display the image.
  create_window(image.sizes());
  display(image);

  // Setup the image segmenter.
  shakti::SegmentationSLIC slic;
  slic.set_distance_weight(1e-4f);

  // Run superpixel segmentation.
  sara::Timer t;
  t.restart();
  Image<int> labels{image.sizes()};
  slic(labels.data(), reinterpret_cast<shakti::Vector4f*>(image.data()),
       image.sizes().data());
  cout << "Segmentation time = " << t.elapsed_ms() << "ms" << endl;

  auto segmentation = Image<Rgba32f>{labels.sizes()};
  auto means = vector<Rgba32f>(labels.flat_array().maxCoeff() + 1, Rgba32f::Zero());
  auto cardinality = vector<int>(labels.flat_array().maxCoeff() + 1, 0);

  for (int y = 0; y < segmentation.height(); ++y)
    for (int x = 0; x < segmentation.width(); ++x)
    {
      means[labels(x, y)] += image(x, y);
      ++cardinality[labels(x, y)];
    }

  for (size_t i = 0; i < means.size(); ++i)
    means[i] /= float(cardinality[i]);

  for (int y = 0; y < segmentation.height(); ++y)
    for (int x = 0; x < segmentation.width(); ++x)
      segmentation(x, y) = means[labels(x, y)];

  // Display segmentation results.
  display(segmentation);
  get_key();
  close_window();
}


void demo_on_video()
{
  auto devices = shakti::get_devices();
  devices.front().make_current_device();
  cout << devices.front() << endl;

  const auto video_filepath = src_path("Segmentation/orion_1.mpg");
  // const string video_filepath = "/home/david/Desktop/test.mp4";
   //const auto video_filepath =
   //    "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4";
  cout << video_filepath << endl;
  VideoStream video_stream{video_filepath};

  auto video_frame_index = int{0};
  auto video_frame = video_stream.frame();

  auto rgba32f_image = Image<Rgba32f>{};
  auto labels = Image<int>{};
  auto segmentation = Image<Rgba32f>{};
  auto means = vector<Rgba32f>{};
  auto cardinality = vector<int>{};

  shakti::SegmentationSLIC slic;
  slic.set_distance_weight(1e-4f);

  while (video_stream.read())
  {
    cout << "[Read frame] " << video_frame_index << "" << endl;

    rgba32f_image = video_frame.convert<Rgba32f>();

    if (!active_window())
      create_window(video_frame.sizes());

    sara::Timer t;
    t.restart();
    labels.resize(video_frame.sizes());
    slic(labels.data(),
         reinterpret_cast<shakti::Vector4f*>(rgba32f_image.data()),
         rgba32f_image.sizes().data());
    cout << "Segmentation time = " << t.elapsed_ms() << "ms" << endl;

    segmentation.resize(video_frame.sizes());
    means = vector<Rgba32f>(labels.flat_array().maxCoeff() + 1, Rgba32f::Zero());
    cardinality = vector<int>(labels.flat_array().maxCoeff() + 1, 0);

    for (int y = 0; y < segmentation.height(); ++y)
    {
      for (int x = 0; x < segmentation.width(); ++x)
      {
        means[labels(x, y)] += rgba32f_image(x, y);
        ++cardinality[labels(x, y)];
      }
    }

    for (size_t i = 0; i < means.size(); ++i)
      means[i] /= float(cardinality[i]);

    for (int y = 0; y < segmentation.height(); ++y)
      for (int x = 0; x < segmentation.width(); ++x)
        segmentation(x, y) = means[labels(x, y)];

    display(segmentation);

    ++video_frame_index;
    cout << endl;
  }
}


GRAPHICS_MAIN()
{
  try
  {
    //demo_on_image();
    demo_on_video();
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }

  return 0;
}
