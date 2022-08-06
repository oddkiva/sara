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

#include <drafts/ChessboardDetection/Corner.hpp>
#include <drafts/ChessboardDetection/NonMaximumSuppression.hpp>

#include "Utilities/ImageOrVideoReader.hpp"


namespace sara = DO::Sara;


auto rotation(const float angle) -> Eigen::Matrix3f
{
  auto T = Eigen::Matrix3f{};
  const Eigen::Matrix2f R = sara::rotation2(angle);
  T.setIdentity();
  T.topLeftCorner<2, 2>() = R;
  return T;
}


auto transform_corners(const std::array<Eigen::Vector2f, 4>& corners,
                       const Eigen::Matrix3f& H)
{
  auto corners_transformed = corners;
  std::transform(corners.begin(), corners.end(), corners_transformed.begin(),
                 [&H](const Eigen::Vector2f& p) {
                   return (H * p.homogeneous()).hnormalized();
                 });
  return corners_transformed;
};

auto axis_aligned_corners(const std::array<Eigen::Vector2f, 4>& corners)
    -> std::array<Eigen::Vector2f, 4>
{
  const auto [xmin_it, xmax_it] = std::minmax_element(
      corners.begin(), corners.end(),
      [](const auto& a, const auto& b) { return a.x() < b.x(); });
  const auto [ymin_it, ymax_it] = std::minmax_element(
      corners.begin(), corners.end(),
      [](const auto& a, const auto& b) { return a.y() < b.y(); });

  const auto xmin = xmin_it->x();
  const auto xmax = xmax_it->x();
  const auto ymin = ymin_it->y();
  const auto ymax = ymax_it->y();

  return std::array{
      Eigen::Vector2f{xmin, ymin},  //
      Eigen::Vector2f{xmax, ymin},
      Eigen::Vector2f{xmax, ymax},
      Eigen::Vector2f{xmin, ymax},
  };
};

auto compute_image_homography_and_sizes(
    const std::array<Eigen::Vector2f, 4>& corners, const Eigen::Matrix3f& H)
    -> std::pair<Eigen::Matrix3f, Eigen::Vector2i>
{
  auto corners_dst = transform_corners(corners, H);
  const auto corners_aa = axis_aligned_corners(corners_dst);
  const auto& top_left = corners_aa[0];
  const auto& bottom_right = corners_aa[2];
  const auto image_dst_sizes = Eigen::Vector2i{
      (bottom_right - top_left).array().round().cast<int>().matrix()};

  // Shift the corners
  auto shift = Eigen::Matrix3f{};
  shift.setIdentity();
  shift.col(2).head(2) = -top_left;

  const auto corners_dst_shifted = transform_corners(corners_dst, shift);

  return std::make_pair(                                   //
      sara::homography(                                    //
          corners[0], corners[1], corners[2], corners[3],  //
          corners_dst_shifted[0], corners_dst_shifted[1],
          corners_dst_shifted[2], corners_dst_shifted[3]),  //
      image_dst_sizes);
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
    const auto sigma = argc < 3 ? 5.f : std::stof(argv[2]);
    const auto cornerness_adaptive_thres =
        argc < 4 ? 1e-5f : std::stof(argv[3]);

    auto video_stream = sara::ImageOrVideoReader{video_file};
    auto video_frame = video_stream.frame();
    auto frame_number = -1;

    const auto w = video_frame.width();
    const auto h = video_frame.height();
    const auto wf = static_cast<float>(video_frame.width());
    const auto hf = static_cast<float>(video_frame.height());
    const auto wh = w * h;
    static const auto corners = std::array{
        Eigen::Vector2f{0, 0},    //
        Eigen::Vector2f{wf, 0},   //
        Eigen::Vector2f{wf, hf},  //
        Eigen::Vector2f{0, hf}    //
    };

    // Prepare the sizes of images.
    auto T_src_to_dst = std::array{
        Eigen::Matrix3f{Eigen::Matrix3f::Identity()},
        rotation(static_cast<float>((45._deg).value)),
        rotation(static_cast<float>((90._deg).value)),
        rotation(static_cast<float>((135._deg).value))  //
    };
    auto T_dst_to_src = T_src_to_dst;
    auto image_dst_sizes = std::array<Eigen::Vector2i, 4>{};
    for (auto i = 0u; i < T_dst_to_src.size(); ++i)
    {
      std::tie(T_src_to_dst[i], image_dst_sizes[i]) =
          compute_image_homography_and_sizes(corners, T_src_to_dst[i]);
      T_dst_to_src[i] = T_src_to_dst[i].inverse();
    }

    auto f = sara::Image<float>{video_frame.sizes()};

    auto f_transformed = std::array<sara::Image<float>, 4>{};
    std::transform(image_dst_sizes.begin(), image_dst_sizes.end(),
                   f_transformed.begin(),
                   [](const auto& sizes) { return sara::Image<float>{sizes}; });

    auto f_filtered = std::array{
        sara::Image<float>{video_frame.sizes()},
        sara::Image<float>{video_frame.sizes()},
        sara::Image<float>{video_frame.sizes()},
        sara::Image<float>{video_frame.sizes()},
    };
    std::for_each(f_filtered.begin(), f_filtered.end(),
                  [w, h](auto& im) { im.resize(w, h); });

    auto fc = sara::Image<float>{video_frame.sizes()};

#ifdef DEBUG_ME
    auto windows = std::array{
        sara::create_window(f_transformed[0].sizes(), std::to_string(0)),
        sara::create_window(f_transformed[1].sizes(), std::to_string(1)),
        sara::create_window(f_transformed[2].sizes(), std::to_string(2)),
        sara::create_window(f_transformed[3].sizes(), std::to_string(3)),
        sara::create_window(video_frame.sizes(), "fc")  //
    };
#else
    sara::create_window(video_frame.sizes(), video_file);
#endif

    static const auto kernel_1d = sara::make_gaussian_kernel(sigma);

    while (video_stream.read())
    {
      ++frame_number;
      if (frame_number % 3 != 0)
        continue;
      SARA_CHECK(frame_number);

      sara::from_rgb8_to_gray32f(video_frame, f);
      f = f.compute<sara::Gaussian>(1.f);

      for (auto i = 0; i < 4; ++i)
      {
        // Warp.
        if (i == 0)
          f_transformed[i] = f;
        else
          sara::warp(f, f_transformed[i], T_dst_to_src[i]);

        // Blur
        auto temp = f_transformed[i];
        sara::apply_row_based_filter(f_transformed[i], temp, kernel_1d.data(),
                                     kernel_1d.size());
        temp.swap(f_transformed[i]);

        // Warp back to original image domain.
        sara::warp(f_transformed[i], f_filtered[i], T_src_to_dst[i]);
      }

      fc.flat_array().fill(0.f);
#pragma omp parallel for
      for (auto xy = 0; xy < wh; ++xy)
      {
        const auto y = xy / w;
        const auto x = xy - y * w;

        auto fc_max = f_filtered[0](x, y);
        for (auto i = 1; i < 4; ++i)
          fc_max = std::max(fc_max, f_filtered[i](x, y));

        auto fc_min = f_filtered[0](x, y);
        for (auto i = 1; i < 4; ++i)
          fc_min = std::min(fc_min, f_filtered[i](x, y));

        fc(x, y) = sara::square(fc_max - fc_min);
      }
      fc = fc.compute<sara::Gaussian>(1.f);

      sara::tic();
      static constexpr auto radius = 10;
      auto corners_int = select(fc, cornerness_adaptive_thres, radius);
      sara::nms(corners_int, fc.sizes(), radius);
      sara::toc("Corner detection");

#ifdef DEBUG_ME
      for (auto i = 0; i < 5; ++i)
      {
        sara::set_active_window(windows[i]);
        if (i < 4)
        {
          sara::display(f_transformed[i]);
          sara::display(f_filtered[i]);
        }
        else
          sara::display(sara::color_rescale(fc));
      }
#endif

      sara::display(f);
      for(const auto& corner: corners_int)
      {
        const auto p = corner.coords;
        sara::fill_circle(p.x(), p.y(), 2, sara::Red8);
      }

      while (sara::any_get_key() != sara::KEY_ESCAPE)
        ;
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
