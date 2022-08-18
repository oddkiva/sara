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

#pragma once

#include <drafts/ChessboardDetection/ChessboardDetector.hpp>


namespace DO::Sara {


  template <typename T, int M, int N>
  inline auto is_nan(const Eigen::Matrix<T, M, N>& x) -> bool
  {
    return std::any_of(x.data(), x.data() + x.size(),
                       [](const T& x) { return std::isnan(x); });
  }

  class ChessboardCorners
  {
  public:
    inline ChessboardCorners() = default;

    inline ChessboardCorners(
        const ChessboardDetector::OrderedChessboardCorners& corners,
        const Length& square_size,
        const std::optional<Eigen::Vector2i>& actual_sizes = std::nullopt)
      : _corners{corners}
      , _square_size{square_size}
      , _actual_sizes{actual_sizes}
    {
    }

    inline auto height() const -> int
    {
      return static_cast<int>(_corners.size());
    }

    inline auto width() const -> int
    {
      if (_corners.empty())
        return 0;
      return static_cast<int>(_corners.front().size());
    }

    inline auto sizes() const -> Eigen::Vector2i
    {
      return {width(), height()};
    }

    inline auto corner_count() const -> int
    {
      return height() * width();
    }

    inline auto empty() const -> bool
    {
      return _corners.empty();
    }

    inline auto actual_sizes() const -> const std::optional<Eigen::Vector2i>&
    {
      return _actual_sizes;
    }

    inline auto image_point(const int x, const int y) const -> const Vector2f&
    {
      return _corners[y][x];
    }

    inline auto scene_point(const int x, const int y) const -> Vector3d
    {
      const auto size = static_cast<double>(_square_size.value);
      return {size * x, size * y, 0};
    }

    inline auto origin() const -> Eigen::Vector2i
    {
      const auto h = height();
      const auto w = width();
      for (auto y = 0; y < h; ++y)
        for (auto x = 0; x < w; ++x)
          if (!is_nan(image_point(x, y)))
            return {x, y};

      return {-1, -1};
    }

    inline auto square_size() const -> const Length&
    {
      return _square_size;
    }

  private:
    ChessboardDetector::OrderedChessboardCorners _corners;
    Length _square_size;
    std::optional<Eigen::Vector2i> _actual_sizes;
  };


  // From:
  // https://stackoverflow.com/questions/40629345/fill-array-dynamicly-with-gradient-color-c
  auto rainbow_color(const float ratio) -> Rgb8;

  auto draw_chessboard(ImageView<Rgb8>& image, const ChessboardCorners& corners)
      -> void;

  auto inspect(ImageView<Rgb8>& image,            //
               const ChessboardCorners& corners,  //
               const Eigen::Matrix3d& K,          //
               const Eigen::Matrix3d& R,          //
               const Eigen::Vector3d& t,          //
               bool pause = false) -> void;

}  // namespace DO::Sara
