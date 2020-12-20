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


GRAPHICS_MAIN()
{
  auto devices = shakti::get_devices();
  devices.front().make_current_device();
  cout << devices.front() << endl;

  constexpr auto use_low_resolution_video = false;
  constexpr auto video_filepath =
      use_low_resolution_video
          ?
          // Video sample with image sizes (320 x 240).
          src_path("Segmentation/orion_1.mpg")
          :
          // Video samples with image sizes (1920 x 1080).
#ifdef _WIN32
          "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"
#else
          "/home/david/Desktop/GOPR0542.MP4"
#endif
      ;
  std::cout << video_filepath << std::endl;
  VideoStream video_stream{video_filepath};

  auto video_frame_index = int{0};
  auto video_frame = video_stream.frame();

  auto rgba32f_image = Image<Rgba32f>{};
  auto labels = Image<int>{};
  auto segmentation = Image<Rgba32f>{};
  auto means = vector<Rgba32f>{};
  auto cardinality = vector<int>{};

  const auto cluster_sizes = shakti::Vector2i{32, 30};
  shakti::SegmentationSLIC slic{cluster_sizes};
  slic.set_distance_weight(1e-4f);

  while (video_stream.read())
  {
    cout << "[Read frame] " << video_frame_index << "" << endl;

    rgba32f_image = video_frame.convert<Rgba32f>();

    if (!active_window())
      set_active_window(create_window(rgba32f_image.sizes()));

    sara::Timer t;
    t.restart();
    labels.resize(rgba32f_image.sizes());
    slic(labels.data(),
         reinterpret_cast<shakti::Vector4f*>(rgba32f_image.data()),
         rgba32f_image.sizes().data());
    cout << "Segmentation time = " << t.elapsed_ms() << "ms" << endl;

    segmentation.resize(rgba32f_image.sizes());
    means = vector<Rgba32f>(labels.flat_array().maxCoeff() + 1, Rgba32f::Zero());
    cardinality = vector<int>(labels.flat_array().maxCoeff() + 1, 0);

    // @todo: port this in CUDA.
    for (int y = 0; y < segmentation.height(); ++y)
    {
      for (int x = 0; x < segmentation.width(); ++x)
      {
        means[labels(x, y)] += rgba32f_image(x, y);
        ++cardinality[labels(x, y)];
      }
    }

    // @todo: port this in CUDA.
    for (size_t i = 0; i < means.size(); ++i)
      means[i] /= float(cardinality[i]);

    // @todo: port this in CUDA.
    for (int y = 0; y < segmentation.height(); ++y)
      for (int x = 0; x < segmentation.width(); ++x)
        segmentation(x, y) = means[labels(x, y)];

    display(segmentation);

    ++video_frame_index;
    cout << endl;
  }

  return 0;
}
