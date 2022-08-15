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

//! @file

#include <drafts/Calibration/Chessboard.hpp>


namespace DO::Sara {

  auto rainbow_color(const float ratio) -> Rgb8
  {
    static constexpr auto quantization = 256;
    static constexpr auto region_count = 6;

    // We want to normalize the ratio so that it fits in to 6 regions where each
    // region is 256 units long.
    const auto normalized =
        static_cast<int>(ratio * quantization * region_count);

    // Find the region for this position.
    const auto region = normalized / quantization;

    // Find the distance to the start of the closest region.
    const auto x = normalized % quantization;

    auto r = std::uint8_t{};
    auto g = std::uint8_t{};
    auto b = std::uint8_t{};

    switch (region)
    {
      // clang-format off
    case 0: r = 255; g = 0;   b = 0;   g += x; break;
    case 1: r = 255; g = 255; b = 0;   r -= x; break;
    case 2: r = 0;   g = 255; b = 0;   b += x; break;
    case 3: r = 0;   g = 255; b = 255; g -= x; break;
    case 4: r = 0;   g = 0;   b = 255; r += x; break;
    case 5: r = 255; g = 0;   b = 255; b -= x; break;
      // clang-format on
    }

    return {r, g, b};
  }

  auto draw_chessboard(ImageView<Rgb8>& image, const ChessboardCorners& corners)
      -> void
  {
    const auto& pattern_size = corners.sizes();
    const auto& w = pattern_size.x();
    const auto& h = pattern_size.y();

    auto c0 = Eigen::Vector2f{};
    auto c1 = Eigen::Vector2f{};
    for (auto y = 0; y < h; ++y)
    {
      const auto ratio = static_cast<float>(y) / h;
      const auto color = rainbow_color(ratio);

      for (auto x = 0; x < w; ++x)
      {
        c0 = c1;
        c1 = corners.image_point(x, y);
        draw_circle(image, c1, 3.f, color, 2);

        if (x == 0 && y == 0)
          continue;
        draw_arrow(image, c0, c1, color, 2);
      }
    }
  }

  auto inspect(ImageView<Rgb8>& image,            //
               const ChessboardCorners& corners,  //
               const Eigen::Matrix3d& K,          //
               const Eigen::Matrix3d& R,          //
               const Eigen::Vector3d& t,          //
               bool pause) -> void
  {
    auto Hr = Eigen::Matrix3d{};
    Hr.col(0) = R.col(0);
    Hr.col(1) = R.col(1);
    Hr.col(2) = t;
    Hr = (K * Hr).normalized();

    const auto ori = corners.origin();
    if (ori.x() == -1 && ori.y() == -1)
    {
      SARA_DEBUG << "Invalid chessboard!" << std::endl;
      return;
    }
    const auto o = corners.image_point(ori.x(), ori.y());
    const auto i = corners.image_point(ori.x() + 1, ori.y());
    const auto j = corners.image_point(ori.x(), ori.y() + 1);
    const auto s = corners.square_size().value;

    const Eigen::Vector3d k3 = R * Eigen::Vector3d::UnitZ() * s + t;
    const Eigen::Vector2f k = (K * k3).hnormalized().cast<float>();

    static const auto red = Rgb8{167, 0, 0};
    static const auto green = Rgb8{89, 216, 26};
    draw_arrow(image, o, i, red, 6);
    draw_arrow(image, o, j, green, 6);
    draw_arrow(image, o, k, Blue8, 6);

    const auto h = corners.height();
    const auto w = corners.width();
    for (auto y = 0; y < h; ++y)
    {
      for (auto x = 0; x < w; ++x)
      {
        const Eigen::Vector3d P = corners.scene_point(x, y);
        const Eigen::Vector2f p1 = corners.image_point(x, y);
        const Eigen::Vector2f p2 = (Hr * P).hnormalized().cast<float>();

        draw_circle(image, p1, 3.f, Cyan8, 3);
        draw_circle(image, p2, 3.f, Magenta8, 3);
        if (pause)
        {
          display(image);
          get_key();
        }
      }
    }
  }

}  // namespace DO::Sara
