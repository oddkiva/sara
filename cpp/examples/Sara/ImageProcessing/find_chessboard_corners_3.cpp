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

#include <map>
#include <unordered_map>

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/EdgeShapeStatistics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>
#include <DO/Sara/VideoIO.hpp>


#include "Chessboard/Erode.hpp"
#include "Chessboard/NonMaximumSuppression.hpp"


namespace sara = DO::Sara;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}


template <typename T>
struct Corner
{
  Eigen::Vector2<T> coords;
  float score;
  auto position() const -> const Eigen::Vector2i&
  {
    return coords;
  }
  auto operator<(const Corner& other) const -> bool
  {
    return score < other.score;
  }
};

// Select the local maxima of the cornerness functions.
auto select(const sara::ImageView<float>& cornerness,
            const float cornerness_adaptive_thres) -> std::vector<Corner<int>>
{
  const auto extrema = sara::local_maxima(cornerness);

  const auto cornerness_max = cornerness.flat_array().maxCoeff();
  const auto cornerness_thres = cornerness_adaptive_thres * cornerness_max;

  auto extrema_filtered = std::vector<Corner<int>>{};
  extrema_filtered.reserve(extrema.size());
  for (const auto& p : extrema)
    if (cornerness(p) > cornerness_thres)
      extrema_filtered.push_back({p, cornerness(p)});
  return extrema_filtered;
};


auto __main(int argc, char** argv) -> int
{
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

  // Harris cornerness parameters.
  //
  // Blur parameter before gradient calculation.
  const auto sigma_D = argc < 3 ? 0.8f : std::stof(argv[2]);
  // Integration domain of the second moment.
  const auto sigma_I = argc < 4 ? 3.f : std::stof(argv[3]);
  // Threshold parameter.
  const auto kappa = argc < 5 ? 0.04f : std::stof(argv[4]);
  const auto cornerness_adaptive_thres = argc < 6 ? 1e-5f : std::stof(argv[5]);

  // Corner filtering.
  const auto nms_radius = argc < 7 ? 10 : std::stoi(argv[6]);
  static constexpr auto downscale_factor = 2;

  // Edge detection.
  static constexpr auto high_threshold_ratio = static_cast<float>(4._percent);
  static constexpr auto low_threshold_ratio =
      static_cast<float>(high_threshold_ratio / 2.);
  using sara::operator""_deg;
  static constexpr auto angular_threshold = static_cast<float>((20._deg).value);

  auto ed = sara::EdgeDetector{{
      high_threshold_ratio,  //
      low_threshold_ratio,   //
      angular_threshold      //
  }};


  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto frame_number = -1;

  auto frame_gray = sara::Image<float>{video_frame.sizes()};
  auto frame_gray_blurred = sara::Image<float>{video_frame.sizes()};
  auto frame_gray_ds =
      sara::Image<float>{video_frame.sizes() / downscale_factor};
  auto grad_f_norm = sara::Image<float>{video_frame.sizes()};
  auto grad_f_ori = sara::Image<float>{video_frame.sizes()};
  auto segmentation_map = sara::Image<std::uint8_t>{video_frame.sizes()};
  auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};

  while (video_stream.read())
  {
    ++frame_number;
    if (frame_number % 3 != 0)
      continue;
    SARA_CHECK(frame_number);

    if (sara::active_window() == nullptr)
    {
      sara::create_window(video_frame.sizes());
      sara::set_antialiasing();
    }

    sara::tic();
    sara::from_rgb8_to_gray32f(video_frame, frame_gray);
    sara::toc("Grayscale conversion");

    sara::tic();
    sara::apply_gaussian_filter(frame_gray, frame_gray_blurred, 1.f);
    sara::scale(frame_gray_blurred, frame_gray_ds);
    sara::toc("Downscale");

    sara::tic();
    ed(frame_gray_ds);
    sara::toc("Curve detection");

    // Calculate Harris cornerness functions.
    sara::tic();
    const auto cornerness = sara::scale_adapted_harris_cornerness(  //
        frame_gray_ds,                                              //
        sigma_I, sigma_D,                                           //
        kappa                                                       //
    );
    const auto grad_f = frame_gray_ds.compute<sara::Gradient>();
    auto corners_int = select(cornerness, cornerness_adaptive_thres);
    sara::toc("Corner detection");


    sara::tic();
    auto corners = std::vector<Corner<float>>{};
    std::transform(
        corners_int.begin(), corners_int.end(), std::back_inserter(corners),
        [&grad_f, sigma_I](const Corner<int>& c) -> Corner<float> {
          const auto p =
              sara::refine_junction_location_unsafe(grad_f, c.coords, sigma_I);
          return {p, c.score};
        });
    sara::toc("Corner refinement");

    const auto& curves = ed.pipeline.edges_as_list;
    auto display = frame_gray.convert<sara::Rgb8>();
    for (auto label = 0u; label < curves.size(); ++label)
    {
      const auto& curve_simplified = ed.pipeline.edges_simplified[label];
      if (curve_simplified.size() < 2u || sara::length(curve_simplified) < 5)
        continue;
      const auto color = sara::Rgb8(rand() % 255, rand() % 255, rand() % 255);

      for (auto i = 0u; i < curve_simplified.size() - 1; ++i)
      {
        const auto a = curve_simplified[i].cast<float>() * downscale_factor;
        const auto b = curve_simplified[i + 1].cast<float>() * downscale_factor;
        sara::draw_line(display, a, b, color, 2);
      }
    }
    for (const auto& p : corners)
      sara::draw_circle(display, downscale_factor * p.coords.x(),
                        downscale_factor * p.coords.y(), 4, sara::Magenta8);
    sara::draw_text(display, 80, 80, std::to_string(frame_number), sara::White8,
                    60, 0, false, true);

    sara::display(display);
    sara::get_key();
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
