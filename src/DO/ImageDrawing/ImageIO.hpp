// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Core/Image.hpp>
#include <easyexif/exif.h>

namespace DO {

  bool readExifInfo(EXIFInfo& exifInfo, const std::string& filePath);

  void print(const EXIFInfo& exifInfo);

  template <typename T>
  void flip(Image<T>& image, int exifOri)
  {
    // 0: unspecified in EXIF data
    // 1: upper left of image
    // 9: undefined
    if (exifOri == 0 || exifOri == 1 || exifOri == 9)
      return;
    
    Image<T> tmp;
    int w = image.width();
    int h = image.height();

    // 3: lower right of image
    if (exifOri == 3)
    {
      tmp.resize(image.sizes());
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
          tmp(x,y) = image(w-1-x, h-1-y);
      image = tmp;
    }
    // 6: upper right of image
    if (exifOri == 6)
    {
      tmp.resize(image.height(), image.width());
      // Transpose.
      for (int y = 0; y < image.height(); ++y)
        for (int x = 0; x < image.width(); ++x)
          tmp(y,x) = image(x,y);
      // Reverse rows.
      for (int y = 0; y < tmp.height(); ++y)
        for (int x = 0; x < tmp.width(); ++x)
        {
          int n_x = tmp.width()-1-x;
          if (x >= n_x)
            break;
          std::swap(tmp(x,y), tmp(n_x,y));
        }
      image = tmp;
    }
    // 8: lower left of image
    if (exifOri == 8)
    {
      tmp.resize(image.height(), image.width());
      // Transpose.
      for (int y = 0; y < image.height(); ++y)
        for (int x = 0; x < image.width(); ++x)
          tmp(y,x) = image(x,y);
      // Reverse cols.
      for (int y = 0; y < tmp.height(); ++y)
      {
        int n_y = tmp.height()-1-y;
        if (y >= n_y)
          break;
        for (int x = 0; x < tmp.width(); ++x)
          std::swap(tmp(x,y), tmp(x,n_y));
      }
      image = tmp;
    }
  }

  bool imread(Image<unsigned char>& image, const std::string& filePath);

  bool imread(Image<Rgb8>& image, const std::string& filePath);

  template <typename T>
  bool imread(Image<T>& image, const std::string& filePath)
  {
    Image<Rgb8> rgb8image;
    if (!imread(rgb8image, filePath))
      return false;
    image = rgb8image.convert<T>();
    return true;
  }

} /* namespace DO */