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

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/AdaptiveBinaryThresholding.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/JunctionRefinement.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>
#include <DO/Sara/VideoIO.hpp>

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

  const auto grad_adaptive_thres = argc < 3 ? 2e-2f : std::stof(argv[2]);

  // Corner filtering.
  const auto kappa = argc < 4 ? 0.04f : std::stof(argv[3]);
  const auto cornerness_adaptive_thres = argc < 5 ? 1e-4f : std::stof(argv[4]);
  const auto nms_radius = argc < 6 ? 2 : std::stoi(argv[5]);

  auto video_stream = sara::VideoStream{video_file};
  auto video_frame = video_stream.frame();
  auto video_frame_copy = sara::Image<sara::Rgb8>{};
  auto frame_number = -1;

  auto f = sara::Image<float>{video_frame.sizes()};
  auto f_blurred = sara::Image<float>{video_frame.sizes()};
  auto grad_f_norm = sara::Image<float>{video_frame.sizes()};
  auto grad_f_ori = sara::Image<float>{video_frame.sizes()};
  auto segmentation_map = sara::Image<std::uint8_t>{video_frame.sizes()};

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
    SARA_CHECK(frame_number);

    sara::tic();
    sara::from_rgb8_to_gray32f(video_frame, f);
    sara::toc("Grayscale conversion");

#ifdef ADAPTIVE_THRESHOLD
    sara::tic();
    static constexpr auto tolerance_parameter = 0.f;
    sara::gaussian_adaptive_threshold(f, 32.f, 3.f, tolerance_parameter,
                                      segmentation_map);
    sara::toc("Adaptive thresholding");

    sara::display(segmentation_map);
#else
    sara::tic();
    sara::apply_gaussian_filter(f, f_blurred, 1.f, 4.f);
    sara::gradient_in_polar_coordinates(f_blurred, grad_f_norm, grad_f_ori);
    const auto grad_max = grad_f_norm.flat_array().maxCoeff();
    const auto grad_thres = grad_adaptive_thres * grad_max;
    auto edge_map = sara::suppress_non_maximum_edgels(
        grad_f_norm, grad_f_ori, 2 * grad_thres, grad_thres);
    for (auto e = edge_map.begin(); e != edge_map.end(); ++e)
      if (*e == 127)
        *e = 0;
    sara::toc("Feature maps");
#endif

    sara::tic();
    const auto M = f_blurred.compute<sara::Gradient>()
                       .compute<sara::SecondMomentMatrix>()
                       .compute<sara::Gaussian>(3.f);
    auto cornerness = sara::Image<float>{f_blurred.sizes()};
    std::transform(M.begin(), M.end(), cornerness.begin(),
                   [kappa](const auto& m) {
                     return m.determinant() - kappa * pow(m.trace(), 2);
                   });
    auto corners = select(cornerness, cornerness_adaptive_thres);
    sara::nms(corners, cornerness.sizes(), nms_radius);
    sara::toc("Corner detection");

    auto disp = edge_map.convert<sara::Rgb8>();
    for (const auto& p : corners)
      sara::fill_circle(disp, p.coords.x(), p.coords.y(), 3, sara::Magenta8);
    sara::display(disp);
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
