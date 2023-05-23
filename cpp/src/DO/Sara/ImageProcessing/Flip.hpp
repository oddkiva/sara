// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Image.hpp>

#if defined(_OPENMP)
#  include <omp.h>
#endif


namespace DO::Sara {

  //! @{
  //! \brief Flip image.
  template <typename T>
  inline void flip_horizontally(Image<T>& image)
  {
    const auto w = image.width();
    const auto h = image.height();

    for (auto y = 0; y < h; ++y)
      for (auto x = 0; x < w / 2; ++x)
        std::swap(image(x, y), image(w - 1 - x, y));
  }

  template <typename T>
  inline void flip_vertically(Image<T>& image)
  {
    const auto w = image.width();
    const auto h = image.height();

    for (auto y = 0; y < h / 2; ++y)
      for (auto x = 0; x < w; ++x)
        std::swap(image(x, y), image(x, h - 1 - y));
  }
  //! @}


  //! @{
  //! \brief Reflect image w.r.t. the image center.
  template <typename T>
  inline void transpose(Image<T>& image)
  {
    auto tmp = Image<T>{};
    const auto w = image.width();
    const auto h = image.height();

    tmp.resize(h, w);

    for (auto y = 0; y < h; ++y)
      for (auto x = 0; x < w; ++x)
        tmp(y, x) = image(x, y);

    image.swap(tmp);
  }

  template <typename T>
  inline void transverse(Image<T>& image)
  {
    transpose(image);
    rotate_ccw_180(image);
  }
  //! @}


  //! @{
  //! \brief Rotate image counter-clockwise.
  template <typename T>
  inline void rotate_ccw_90(Image<T>& image)
  {
    transpose(image);
    flip_vertically(image);
  }

  template <typename T>
  inline void rotate_ccw_180(Image<T>& image)
  {
    auto tmp = Image<T>{image.sizes()};
    const auto w = image.width();
    const auto h = image.height();

    for (auto y = 0; y < h; ++y)
      for (auto x = 0; x < w; ++x)
        tmp(x, y) = image(w - 1 - x, h - 1 - y);

    image.swap(tmp);
  }

  template <typename T>
  inline void rotate_ccw_270(Image<T>& image)
  {
    transpose(image);
    flip_horizontally(image);
  }
  //! @}


  //! @{
  //! \brief Rotate image clockwise.
  template <typename T>
  inline void rotate_cw_90(Image<T>& image)
  {
    rotate_ccw_270(image);
  }

  template <typename T>
  inline void rotate_cw_180(Image<T>& image)
  {
    rotate_ccw_180(image);
  }

  template <typename T>
  inline void rotate_cw_270(Image<T>& image)
  {
    rotate_ccw_90(image);
  }
  //! @}

  namespace v2 {

    template <typename T>
    inline auto flip_horizontally(const ImageView<T>& src, ImageView<T>& dst)
        -> void
    {
      const auto w = dst.width();
      const auto h = dst.height();
      const auto wh = w * h;

#pragma omp parallel for
      for (auto xy = 0; xy < wh; ++xy)
      {
        const auto y = xy / w;
        const auto x = xy - y * w;
        dst(x, y) = src(w - 1 - x, y);
      }
    }

    template <typename T>
    inline auto flip_vertically(const ImageView<T>& src, ImageView<T>& dst)
        -> void
    {
      const auto w = dst.width();
      const auto h = dst.height();
      const auto wh = w * h;

#pragma omp parallel for
      for (auto xy = 0; xy < wh; ++xy)
      {
        const auto y = xy / w;
        const auto x = xy - y * w;
        dst(x, y) = src(x, h - 1 - y);
      }
    }

    template <typename T>
    inline auto transpose(const ImageView<T>& src, ImageView<T>& dst) -> void
    {
      const auto w = dst.width();
      const auto h = dst.height();
      const auto wh = w * h;

#pragma omp parallel for
      for (auto xy = 0; xy < wh; ++xy)
      {
        const auto y = xy / w;
        const auto x = xy - y * w;
        dst(x, y) = src(y, x);
      }
    }

    template <typename T>
    inline auto rotate_cw_90(const ImageView<T>& src, ImageView<T>& dst)
        -> void
    {
      const auto w = dst.width();
      const auto h = dst.height();
      const auto wh = w * h;

#pragma omp parallel for
      for (auto xy = 0; xy < wh; ++xy)
      {
        const auto y = xy / w;
        const auto x = xy - y * w;
        dst(x, y) = src(y, x);
      }
    }

    template <typename T>
    inline auto rotate_ccw_180(const ImageView<T>& src, ImageView<T>& dst)
        -> void
    {
      const auto w = src.width();
      const auto h = src.height();

      const auto wh = w * h;

#pragma omp parallel for
      for (auto xy = 0; xy < wh; ++xy)
      {
        const auto y = xy / w;
        const auto x = xy - y * w;
        dst(x, y) = src(w - 1 - x, h - 1 - y);
      }
    }

    template <typename T>
    inline auto rotate_cw_180(const ImageView<T>& src, ImageView<T>& dst)
        -> void
    {
      rotate_ccw_189(src, dst);
    }

  }  // namespace v2

}  // namespace DO::Sara
