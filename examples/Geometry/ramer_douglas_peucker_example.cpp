// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Core/Timer.hpp>

using namespace std;
using namespace DO::Sara;


GRAPHICS_MAIN()
{
  auto w = 400;
  auto h = 400;

  create_window(w, h);
  set_antialiasing();

  auto circle = vector<Point2d>{};

  const Point2d c{ w / 2., h / 2. };
  const auto r = double{ h * 0.4 };

  const int N{ 1000 };
  for (int i = 0; i < N; ++i)
    circle.push_back(Point2d{
    c + r * Vector2d{ cos(2 * M_PI*i / N), sin(2 * M_PI*i / N) }
  });

  auto simplified_circle = ramer_douglas_peucker(circle, 5.);

  draw_poly(circle, Red8, 2);
  draw_poly(simplified_circle, Green8, 2);
  get_key();


  return 0;
}
