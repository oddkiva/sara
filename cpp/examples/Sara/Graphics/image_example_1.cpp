// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Graphics.hpp>

using namespace std;
using namespace DO::Sara;

GRAPHICS_MAIN()
{
  Window W = create_window(512, 512, "Bitmaps");
  // Array of bytes
  Rgb8 cols[256 * 256];
  // Some RGB function of (i,j)
  for (int j = 0; j < 256; j++)
    for (int i = 0; i < 256; i++)
      cols[i + 256 * j] = Rgb8(i, 255 - i, (j < 128) ? 255 : 0);
  // Draw this 256x256 RGB bitmap in (0,0)
  display(ImageView<Rgb8>{cols, {256, 256}});

  // An array of colors.
  // Color3ub = 3D color vector where each channel has a value in [0,255].
  Rgb8 cols2[256 * 256];
  for (int j = 0; j < 256; j++)
    for (int i = 0; i < 256; i++)
      cols2[i + 256 * j] = Rgb8(i, (2 * j) % 256, (i + j) % 256);
  // Display the bitmap from the following top-left corner (0,256)
  display(ImageView<Rgb8>{cols2, {256, 256}}, {0, 256});

  // A grayscale image.
  std::uint8_t grey[256 * 256];
  for (int j = 0; j < 256; ++j)
    for (int i = 0; i < 256; ++i)
      grey[i + 256 * j] = static_cast<unsigned char>(  //
          128 + 127 * sin((i + j) / 10.));
  // Display the bitmap from the following top-left corner (0,256)
  display(ImageView<std::uint8_t>{grey, {256, 256}}, {256, 0});

  click();
  close_window(W);

  return 0;
}
