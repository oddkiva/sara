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
#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>


namespace DO { namespace Sara {

  DO_SARA_EXPORT
  bool read_exif_info(EXIFInfo& exif_info, const std::string& file_path);

  DO_SARA_EXPORT
  std::ostream& operator<<(std::ostream& os, const EXIFInfo& exifInfo);


  enum ExifOrientationTag
  {
    Unspecified = 0,
    Upright = 1,
    FlippedHorizontally = 2,
    RotatedCCW_180 = 3,
    FlippedVertically = 4,
    Transposed = 5,
    RotatedCCW_90 = 6,
    Transversed = 7,
    RotatedCW_90 = 8,
    Undefined = 9
  };

  template <typename T>
  void make_upright_from_exif(Image<T>& image, unsigned short exif_orientation_tag)
  {
    switch (exif_orientation_tag)
    {
    case Unspecified:           // 0
    case Upright:               // 1
    case Undefined:             // 9
      break;

    case FlippedHorizontally:   // 2
      flip_horizontally(image);
      break;

    case RotatedCCW_180:        // 3
      rotate_ccw_180(image);
      break;

    case FlippedVertically:     // 4
      flip_vertically(image);
      break;

    case Transposed:            // 5
      transpose(image);
      break;

    case RotatedCCW_90:         // 6
      rotate_cw_90(image);
      break;

    case Transversed:           // 7
      transverse(image);
      break;

    case RotatedCW_90:          // 8
      rotate_ccw_90(image);
      break;

    default:
      break;
    }
  }



} /* namespace Sara */
} /* namespace DO */
