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

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>
#include <DO/Sara/VideoIO.hpp>


#include "Chessboard/Erode.hpp"
#include "Chessboard/NonMaximumSuppression.hpp"


namespace sara = DO::Sara;


struct Corner
{
  Eigen::Vector2i coords;
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
            const float cornerness_adaptive_thres) -> std::vector<Corner>
{
  const auto extrema = sara::local_maxima(cornerness);

  const auto cornerness_max = cornerness.flat_array().maxCoeff();
  const auto cornerness_thres = cornerness_adaptive_thres * cornerness_max;

  auto extrema_filtered = std::vector<Corner>{};
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
  const auto sigma_D =
      argc < 3 ? std::sqrt(std::pow(1.6f, 2.f) - 1) : std::stof(argv[2]);
  // Integration domain of the second moment.
  const auto sigma_I = argc < 4 ? 3.f : std::stof(argv[3]);
  // Threshold parameter.
  const auto kappa = argc < 5 ? 0.04f : std::stof(argv[4]);
  const auto cornerness_adaptive_thres = argc < 6 ? 1e-5f : std::stof(argv[5]);

  // Corner filtering.
  const auto nms_radius = argc < 7 ? 10 : std::stoi(argv[6]);
  static constexpr auto grad_adaptive_thres = 2e-2f;
  static constexpr auto downscale_factor = 2;

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

    if (sara::active_window() == nullptr)
    {
      sara::create_window(video_frame.sizes());
      sara::set_antialiasing();
    }

    sara::tic();
    sara::from_rgb8_to_gray32f(video_frame, frame_gray);
    sara::toc("Grayscale conversion");

    sara::tic();
    sara::apply_gaussian_filter(frame_gray, frame_gray_blurred, 1.2f);
    sara::scale(frame_gray_blurred, frame_gray_ds);
    sara::toc("Downscale");

#ifdef ADAPTIVE_THRESHOLD
    sara::tic();
    static constexpr auto tolerance_parameter = 0.f;
    sara::gaussian_adaptive_threshold(frame_gray, 32.f, 3.f,
                                      tolerance_parameter, segmentation_map);
    sara::toc("Adaptive thresholding");

    sara::tic();
    auto segmentation_map_eroded = segmentation_map;
    for (auto i = 0; i < 1; ++i)
    {
      sara::binary_erode_3x3(segmentation_map, segmentation_map_eroded);
      segmentation_map.swap(segmentation_map_eroded);
    }
    sara::toc("Erosion 3x3");
#else
    sara::tic();
    sara::gradient_in_polar_coordinates(frame_gray_blurred, grad_f_norm,
                                        grad_f_ori);
    const auto grad_max = grad_f_norm.flat_array().maxCoeff();
    const auto grad_thres = grad_adaptive_thres * grad_max;
    auto edge_map = sara::suppress_non_maximum_edgels(
        grad_f_norm, grad_f_ori, 2 * grad_thres, grad_thres);
    std::for_each(edge_map.begin(), edge_map.end(), [](auto& v) {
      if (v != 255)
        v = 0;
    });
    sara::toc("Feature maps");
#endif

    // Calculate Harris cornerness functions.
    sara::tic();
    const auto cornerness = sara::scale_adapted_harris_cornerness(  //
        frame_gray_ds,                                              //
        sigma_I, sigma_D,                                           //
        kappa                                                       //
    );
    auto corners = select(cornerness, cornerness_adaptive_thres);
    sara::nms(corners, cornerness.sizes(), nms_radius);
    sara::toc("Corner detection");

#ifdef ADAPTIVE_THRESHOLD
    display = segmentation_map.convert<sara::Rgb8>();
#else
    display = edge_map.convert<sara::Rgb8>();
#endif
    for (const auto& p : corners)
    {
      sara::fill_circle(display, downscale_factor * p.coords.x(),
                        downscale_factor * p.coords.y(), 4, sara::Magenta8);
      // sara::draw_circle(display, 2 * p.coords.x(), 2 * p.coords.y(), 6,
      //                   sara::Magenta8, 3);
    }
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