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

#pragma once

#include <stdexcept>
#include <unordered_map>

#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/Geometry/Algorithms/Region.hpp>

namespace DO::Sara {

  struct Border
  {
    enum class Type : std::uint8_t
    {
      NonBorder = 0,
      OuterBorder = 1,
      HoleBorder = 2
    };

    int id;
    int parent;
    Type type;
    std::vector<Eigen::Vector2i> curve;
  };

  inline auto is_outer_border_pixel(const ImageView<int>& f,
                                    const Eigen::Vector2i& p) -> bool
  {
    if (p.x() > 0)
    {
      const auto& q = Eigen::Vector2i{p.x() - 1, p.y()};
      return f(p) == 1 && f(q) == 0;
    }

    return false;
  }

  inline auto is_hole_border_pixel(const ImageView<int>& f,
                                   const Eigen::Vector2i& p) -> bool
  {
    if (p.x() < f.width() - 1)
    {
      const auto& q = Eigen::Vector2i{p.x() + 1, p.y()};
      return f(p) >= 1 && f(q) == 0;
    }

    return false;
  }

  inline auto parent_border(const int lnbd, const int newly_found_border,
                            const std::unordered_map<int, Border>& borders)
  {
    if (lnbd == newly_found_border)
      return borders.at(lnbd).parent;
    if (lnbd != newly_found_border)
      return lnbd;
    return -1;
  }

  inline auto follow_border(ImageView<int>& f,
                            std::vector<Eigen::Vector2i>& curve,
                            Eigen::Vector2i p, Eigen::Vector2i p2,
                            const int nbd)
  {
    // clang-format off
    // Directions enumerated in clockwise order.
    static const auto cw_dirs = std::array{
      Vector2i{ 1,  0},  // East
      Vector2i{ 1,  1},  // South-East
      Vector2i{ 0,  1},  // South
      Vector2i{-1,  1},  // South-West
      Vector2i{-1,  0},  // West
      Vector2i{-1, -1},  // North-West
      Vector2i{ 0, -1},  // North
      Vector2i{ 1, -1}   // North-East
    };
    // Directions enumerated in counterclockwise order.
    static const auto ccw_dirs = std::array{
      Vector2i{ 1,  0},  // East
      Vector2i{ 1, -1},  // North-East
      Vector2i{ 0, -1},  // North
      Vector2i{-1, -1},  // North-West
      Vector2i{-1,  0},  // West
      Vector2i{-1,  1},  // South-West
      Vector2i{ 0,  1},  // South
      Vector2i{ 1,  1},  // South-East
    };
    // clang-format on

    const auto in_image_domain = [&f](const Eigen::Vector2i& p) {
      return 0 <= p.x() && p.x() < f.width() &&  //
             0 <= p.y() && p.y() < f.height();
    };

    // Initialize the curve points.
    curve.emplace_back(p);

    auto p1 = Eigen::Vector2i{};
    auto p3 = Eigen::Vector2i{};
    auto p4 = Eigen::Vector2i{};
    auto dir = Eigen::Vector2i{};

    // Step (3.1) Find the first 1-pixel p1 in the 8-neighborhood of p, starting
    // from p2.
    dir = p2 - p;
    auto d = std::find(cw_dirs.begin(), cw_dirs.end(), dir) - cw_dirs.begin();
    auto i = 0;
    for (; i < 8; ++i)
    {
      p1 = p + cw_dirs[(d + i) % 8];
      if (!in_image_domain(p1))
        continue;
      if (f(p1) != 0)
        break;
    }
    if (i == 8)
    {
      // The curve is just a point.
      f(p) = -nbd;
      return;
    }

    // Step (3.2): Loop initialization.
    p2 = p1;  // p2 = the second nonzero border pixel.
    p3 = p;   // p3 = the first  nonzero border pixel.

    while (true)
    {
      // Step (3.3):
      // Scan the 8-neighborhood of the 1-pixel p in counterclockwise order,
      // starting from the 0-pixel right next after p2 (which is a 1-pixel).
      dir = p2 - p3;
      // d is the direction index of p2:
      d = std::find(ccw_dirs.begin(), ccw_dirs.end(), dir) - ccw_dirs.begin();
      // We don't want to start from p2 but from the next 0-pixel (in
      // counterclockwise order):
      d += 1;
      for (i = 0; i < 8; ++i)
      {
        p4 = p3 + ccw_dirs[(d + i) % 8];
        if (!in_image_domain(p4))
          continue;
        if (f(p4) != 0)
          break;
      }

      // Step (3.4): Relabel the border map "f".
      auto fp3 = 0;
      if (p3.x() + 1 < f.width() &&
          f(p3.x() + 1, p3.y()) == 0)  // the "right" side of the border.
        fp3 = -nbd;
      if (f(p3) == 1 &&              //
          p3.x() + 1 < f.width() &&  //
          f(p3.x() + 1, p3.y()) != 0)
        fp3 = nbd;
      // A special case closer to what Suzuki & Abe illustrates in the examples.
      // The sign is still wrong.
      if (f(p3) == 1 &&                                 //
          0 <= p3.x() - 1 && p3.x() + 1 < f.width() &&  //
          f(p3.x() - 1, p3.y()) == 0 &&                 //
          f(p3.x() + 1, p3.y()) == 0)
        fp3 = nbd;
      if (fp3 != 0)
        f(p3) = fp3;


      // Step (3.5): Termination criterion: is the curve closed now?
      if (p4 == p && p3 == p1)
        break;
      // Go back to Step (3.3).
      p2 = p3;
      p3 = p4;
      curve.emplace_back(p3);
    }
  }

  inline auto suzuki_abe_follow_border(
      const ImageView<std::uint8_t>& binary_segmentation_map)
      -> std::unordered_map<int, Border>
  {
    auto f = Image<int>{binary_segmentation_map.sizes()};
    std::transform(binary_segmentation_map.begin(),
                   binary_segmentation_map.end(), f.begin(),
                   [](const auto& v) { return v != 0 ? 1 : 0; });

    const auto w = f.width();
    const auto h = f.height();

    auto borders = std::unordered_map<int, Border>{};

    auto nbd = 1;  // The sequential number of the current border is 1,
                   // because the image frame is a hole border.
    auto lnbd = 1;

    for (auto y = 0; y < h; ++y)
    {
      lnbd = 1;

      for (auto x = 0; x < w; ++x)
      {
        // Step (1.a)
        const auto p = Eigen::Vector2i{x, y};
        if (is_outer_border_pixel(f, p))
        {
          ++nbd;
          const auto p2 = Eigen::Vector2i{x - 1, y};

          // Step (2)
          auto& border = borders[nbd];
          border.id = nbd;
          border.parent = parent_border(lnbd, nbd, borders);
          border.type = Border::Type::OuterBorder;

          // Step (3)
          follow_border(f, border.curve, p, p2, nbd);

// #  define DEBUG_ME
#ifdef DEBUG_ME
          SARA_CHECK(static_cast<int>(border.type));
          SARA_CHECK(nbd);
          SARA_CHECK(lnbd);
          std::cout << "border map" << std::endl;
          std::cout << f.matrix() << std::endl;
#endif
        }
        // Step (1.b)
        else if (is_hole_border_pixel(f, p))
        {
#ifdef DEBUG_ME
          SARA_DEBUG << "HOLE BORDER" << std::endl;
#endif
          ++nbd;
          const auto p2 = Eigen::Vector2i{x + 1, y};
          if (f(p) != 1)
            lnbd = f(p);

          auto& border = borders[nbd];
          // Find the parent of the border.
          border.id = nbd;
          border.parent = parent_border(lnbd, nbd, borders);
          border.type = Border::Type::HoleBorder;

          // Step (3)
          follow_border(f, border.curve, p, p2, nbd);

#ifdef DEBUG_ME
          SARA_CHECK(static_cast<int>(border.type));
          SARA_CHECK(nbd);
          SARA_CHECK(lnbd);
          std::cout << "border map" << std::endl;
          std::cout << f.matrix() << std::endl;
#endif
        }

        // Step (1.c): goto Step (4).
        if (f(p) != 1 && f(p) != 0)
          lnbd = std::abs(f(p));
      }
    }

    return borders;
  }

}  // namespace DO::Sara
