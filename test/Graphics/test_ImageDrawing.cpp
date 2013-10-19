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

using namespace DO;

struct Bresenham
{
  template <typename Color>
  static inline void putColor(Image<Color>& image, int x, int y,
                              const Color& color)
  {
    if (x < 0 || x >= image.width() || y < 0 || y >= image.height())
      return;
    image(x,y) = color;
  }

  template <typename Color>
  static void drawLine(Image<Color>& image, int x0, int y0, int x1, int y1,
                       const Color& color)
  {
    const int dx = abs(x1-x0);
    const int dy = abs(y1-y0); 
    const int sx = x0 < x1 ? 1 : -1;
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx-dy;

    while (true)
    {
      // Put color to image at current point $(x_0, y_0)$
      putColor(image, x0, y0, color);

      // Stop drawing when we reach the end point $(x_1, y_1)$
      if (x0 == x1 && y0 == y1)
        return;
      const int e2 = 2*err;
      if (e2 > -dy)
      {
        err -= dy;
        x0 += sx;
      }

      // Stop drawing when we reach the end point $(x_1, y_1)$
      if (x0 == x1 && y0 == y1)
      {
        putColor(image, x0, y0, color);
        return;
      }
      if (e2 < dx)
      {
        err += dx;
        y0 += sy;
      }
    }
  }

  template <typename Color>
  static void drawCircle(Image<Color>& image, int x1, int y1, int r,
                         const Color& color)
  {

  }

  template <typename Color>
  static void drawEllipse(Image<Color>& image, int x1, int y1, int r1, int r2,
                          const Color& color)
  {

  }
};

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

struct Wu
{
  template <typename T>
  void drawLine(Image<T>& image, int x1, int y1, int x2, int y2, const T& Color)
  {

  }

  template <typename T>
  void drawCircle(Image<T>& image, int x1, int y1, int r, const T& Color)
  {

  }

  template <typename T>
  void drawEllipse(Image<T>& image, int x1, int y1, int r1, int r2, const T& Color)
  {

  }
};
