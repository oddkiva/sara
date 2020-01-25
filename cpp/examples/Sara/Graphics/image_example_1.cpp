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
  Color3ub cols[256*256];
  // Some RGB function of (i,j)
  for (int j = 0; j < 256; j++)
    for (int i = 0; i < 256; i++)
      cols[i+256*j]= Color3ub(i, 255-i, (j<128)?255:0);
  // Draw this 256x256 RGB bitmap in (0,0)
  put_color_image(0, 0, cols, 256, 256);

  // An array of colors.
  // Color3ub = 3D color vector where each channel has a value in [0,255].
  Color3ub cols2[256*256];
  for (int j = 0; j < 256; j++)
    for (int i = 0; i < 256; i++)
      cols2[i+256*j]=Color3ub(i, (2*j)%256, (i+j)%256);  // RGB colors.
  // Display the bitmap from the following top-left corner (0,256)
  // TODO: rename this function.
  put_color_image(Point2i(0, 256), cols2, 256, 256);

  // A grayscale image.
  unsigned char grey[256*256];
  for (int j = 0; j < 256; ++j)
    for (int i = 0; i < 256; ++i)
      grey[i+256*j] = static_cast<unsigned char>(128+127*sin((i+j)/10.));
  // Display the bitmap from the following top-left corner (0,256)
  // TODO: rename this function.
  put_grey_image(256 ,0, grey, 256, 256); // Draw at point (256,0);

  click();
  close_window(W);

  return 0;
}
