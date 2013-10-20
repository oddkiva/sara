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

#include <DO/Graphics.hpp>
#include "ImageDrawing.hpp"

using namespace DO;


int main()
{
  Image<Rgb8> img(300, 300);
  img.array().fill(White8);

  const float max_slope_value = 50.f;
  // Check line drawing when "start point < end point"
  for (float i = 0; i < max_slope_value; i += 0.2f)
    Bresenham::drawLine(img, 10, 10, 290, 290*i+10, Black8);
  // Check line drawing when "start point > end point"
  for (float i = 0.1f; i < max_slope_value; i += 0.2f)
    Bresenham::drawLine(img, 290, 290*i+10, 10, 10, Red8);

  viewImage(img);

  return 0;
}