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
  // Open a 300x200 window.
  Window W = create_window(300, 200);
  set_antialiasing(active_window());
  set_transparency(active_window());

  draw_point(Point2f(10.5f, 10.5f), Green8);
  draw_point(Point2f(20.8f, 52.8132f), Green8);

  draw_line(Point2f(10.5f, 10.5f), Point2f(20.8f, 52.8132f), Blue8, 2);
  draw_line(Point2f(10.5f, 20.5f), Point2f(20.8f, 52.8132f), Magenta8, 5);

  // Draw an oriented ellipse with:
  // center = (150, 100)
  // r1 = 10
  // r2 = 20
  // orientation = 45°
  // in cyan color, and a pencil width = 1.
  draw_ellipse(Point2f(150.f, 100.f), 10.f, 20.f, 45.f, Cyan8, 1);
  draw_ellipse(Point2f(50.f, 50.f), 10.f, 20.f, 0.f, Red8, 1);

  fill_circle(Point2f(100.f, 100.f), 10.f, Blue8);
  fill_ellipse(Point2f(150.f, 150.f), 10.f, 20.f, 72.f, Green8);

  Point2f p1(rand()%300, rand()%200);
  Point2f p2(rand()%300, rand()%200);
  draw_point((p1*2+p2)/2, Green8);

  click();
  close_window(W);

  return 0;
}
