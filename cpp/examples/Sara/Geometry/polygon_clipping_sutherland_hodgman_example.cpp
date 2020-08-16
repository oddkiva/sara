// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Core/Timer.hpp>

using namespace std;
using namespace DO::Sara;


GRAPHICS_MAIN()
{
  const auto w = 400;
  const auto h = 400;

  auto timer = Timer{};
  auto elapsed = double{};

  create_window(w, h);
  set_antialiasing();

  std::vector<Point2d> poly, clip, res;
  {
    int step = 18;
    for (int i = 0; i < step; ++i)
    {
      auto p = Point2d{
        w / 2. + 100 * cos(i * 2 * M_PI / step),
        h / 2. + 150 * sin(i * 2 * M_PI / step)
      };
      poly.push_back(p);

      p.array() += 90;
      clip.push_back(p);
    }
  }
  draw_polygon(poly, Red8);
  draw_polygon(clip, Blue8);
  get_key();

  const auto num_iter = 1000;
  timer.restart();
  for (auto i = 0; i < num_iter; ++i)
    res = sutherland_hodgman(poly, clip);
  elapsed = timer.elapsed_ms() / num_iter;
  cout << "Intersection computation time = " << elapsed << " milliseconds"
       << endl;

  draw_polygon(res, Green8,5);
  get_key();

  return 0;
}
