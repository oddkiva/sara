// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <easyexif/exif.h>

#include <DO/Sara/Defines.hpp>


namespace DO { namespace Sara {

  DO_SARA_EXPORT
  bool read_exif_info(EXIFInfo& exif_info, const std::string& file_path);

  // The top-left corner is mapped to 
  enum ExifOrientationTag
  {
    Unspecified = 0,
    Upright = 1
    MirrorHorizontal = 2,
    Rotate180 = 3,
    Vertical 4,
    Transpose = 5,
    Rotate90 = 6,
    Transverse = 7,
    Rotate270 = 8,
    Undefined = 9
  };

  template <typename T>
  void flip_from_exif_orientation_tag(Image<T>& image,
                                      int exif_orientation_code)
  {
    // 0: unspecified in EXIF data
    // 1: upper left of image
    // 9: undefined
    if (exif_orientation_code == 0 ||
        exif_orientation_code == 1 ||
        exif_orientation_code == 9)
      return;

    auto tmp = Image<T>{};
    const auto w = image.width();
    const auto h = image.height();

    // 3: lower right of image
    if (exif_orientation_code == 3)
    {
      tmp.resize(w, h);
      for (auto y = 0; y < h; ++y)
        for (auto x = 0; x < w; ++x)
          tmp(x, y) = image(w - 1 - x, h - 1 - y);
      image.swap(tmp);
    }
    // 6: upper right of image
    if (exif_orientation_code == 6)
    {
      tmp.resize(h, w);
      // Transpose.
      for (auto y = 0; y < image.height(); ++y)
        for (auto x = 0; x < image.width(); ++x)
          tmp(y,x) = image(x,y);

      // Reverse rows.
      for (auto y = 0; y < tmp.height(); ++y)
        for (auto x = 0; x < tmp.width(); ++x)
        {
          int n_x = tmp.width()-1-x;
          if (x >= n_x)
            break;
          std::swap(tmp(x,y), tmp(n_x,y));
        }
      image.swap(tmp);
    }
    // 8: lower left of image
    if (exif_orientation_code == 8)
    {
      tmp.resize(h, w);

      // Transpose.
      for (auto y = 0; y < image.height(); ++y)
        for (auto x = 0; x < image.width(); ++x)
          tmp(y,x) = image(x,y);

      // Reverse cols.
      for (auto y = 0; y < tmp.height(); ++y)
      {
        auto n_y = tmp.height() - 1 - y;
        if (y >= n_y)
          break;
        for (auto x = 0; x < tmp.width(); ++x)
          std::swap(tmp(x,y), tmp(x,n_y));
      }
      image.swap(tmp);
    }
  }

  DO_SARA_EXPORT
  std::ostream& operator<<(std::ostream& os, const EXIFInfo& exifInfo);


} /* namespace Sara */
} /* namespace DO */
