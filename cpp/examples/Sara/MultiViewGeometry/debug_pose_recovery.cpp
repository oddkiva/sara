// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //


#include <DO/Sara/Graphics.hpp>

#include <DO/Sara/Logging/Logger.hpp>
#include <DO/Sara/SfM/Odometry/OdometryPipeline.hpp>

#include <string_view>

#if defined(_OPENMP)
#  include <omp.h>
#endif


namespace fs = std::filesystem;
namespace sara = DO::Sara;


class SingleWindowApp
{
public:
  SingleWindowApp(const std::string& title)
    : _title{title}
  {
  }

  ~SingleWindowApp()
  {
    sara::close_window();
  }

  auto set_config(const fs::path& video_path,
                  const sara::v2::BrownConradyDistortionModel<double>& camera)
      -> void
  {
    _pipeline.set_config(video_path, camera);
  }

  auto run() -> void
  {
    const auto sizes = _pipeline._video_streamer.frame_rgb8().sizes();
    sara::create_window(sizes, _title);

    auto frames = std::vector<sara::Image<sara::Rgb8>>{};

    while (true)
    {
      if (!_pipeline.read())
        break;

      if (!_pipeline._video_streamer.skip())
      {
        _pipeline.process();

        const auto frame_annotated = _pipeline.make_display_frame();
        frames.emplace_back(frame_annotated);
        sara::display(frames.back());

        sara::get_key();
      }
    }
  }

private:
  std::string _title;
  sara::OdometryPipeline _pipeline;
};


auto graphics_main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
    -> int
{
#if defined(_OPENMP)
  const auto num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
  Eigen::setNbThreads(num_threads);
#endif

#define USE_HARDCODED_VIDEO_PATH
#if defined(USE_HARDCODED_VIDEO_PATH) && defined(__APPLE__)
  const auto video_path =
      fs::path{"/Users/oddkiva/Desktop/datasets/sample-1.mp4"};
  if (!fs::exists(video_path))
  {
    fmt::print("Video {} does not exist", video_path.string());
    return EXIT_FAILURE;
  }
#else
  if (argc < 2)
  {
    std::cout << fmt::format("Usage: {} VIDEO_PATH\n",
                             std::string_view{argv[0]});
    return EXIT_FAILURE;
  }

  const auto video_path = fs::path{argv[1]};
#endif

  auto camera = sara::v2::BrownConradyDistortionModel<double>{};
  {
    camera.fx() = 917.2878392016245;
    camera.fy() = 917.2878392016245;
    camera.shear() = 0.;
    camera.u0() = 960.;
    camera.v0() = 540.;
    // clang-format off
    camera.k() <<
      -0.2338367557617234,
      +0.05952465745165465,
      -0.007947847982157091;
    // clang-format on
    camera.p() << -0.0003137658969742134, 0.00021943576376532096;
  }

  try
  {
    auto app = SingleWindowApp{"Odometry: " + video_path.string()};
    app.set_config(video_path, camera);
    app.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(graphics_main);
  return app.exec();
}
