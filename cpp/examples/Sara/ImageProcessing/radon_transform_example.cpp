// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <omp.h>

#include <array>

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include "Utilities/ImageOrVideoReader.hpp"


namespace sara = DO::Sara;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}

auto __main(int argc, char** argv) -> int
{
  try
  {
    using sara::operator""_deg;

    omp_set_num_threads(omp_get_max_threads());

#ifdef _WIN32
    const auto video_file = sara::select_video_file_from_dialog_box();
    if (video_file.empty())
      return 1;
#else
    if (argc < 2)
      return 1;
    const auto video_file = std::string{argv[1]};
#endif

    auto video_stream = sara::ImageOrVideoReader{video_file};
    auto video_frame = video_stream.frame();
    auto frame_number = -1;

    auto f = sara::Image<sara::Rgb8>{video_frame.width(), video_frame.width()};
    auto fs = std::array{
     sara::Image<sara::Rgb8>{video_frame.width(), video_frame.width()},
     sara::Image<sara::Rgb8>{video_frame.height(), video_frame.width()},
    };

    auto T = Eigen::Matrix3f{};
    const Eigen::Vector2f c = video_frame.sizes().cast<float>() * 0.5f;
    const Eigen::Matrix2f R = sara::rotation2<float>(45._deg);
    T.setIdentity();
    T.topLeftCorner<2, 2>() = R;
    T.col(2).head(2) = c - R * c;

    auto T2 = Eigen::Matrix3f{};
    const Eigen::Vector2f c2 = fs[1].sizes().cast<float>() * 0.5f;
    const Eigen::Matrix2f R2 = sara::rotation2<float>(90._deg);
    T2.setIdentity();
    T2.topLeftCorner<2, 2>() = R2;
    T2.col(2).head(2) = c2 - R2 * c2;

    while (video_stream.read())
    {
      ++frame_number;
      if (frame_number % 3 != 0)
        continue;
      SARA_CHECK(frame_number);

      if (sara::active_window() == nullptr)
      {
        sara::create_window(fs[1].sizes(), video_file);
        sara::set_antialiasing();
      }

      sara::warp(video_frame, fs[1], T2);

      sara::display(fs[1]);
      sara::toc("Display");
      sara::get_key();
    }
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
