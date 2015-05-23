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
using namespace DO;

GRAPHICS_MAIN()
{
  int w, h;
  w = h = 400;

  Timer timer;
  double elapsed;

  create_window(w,h);
  set_antialiasing();

  std::vector<Point2d> poly, clip, res;
  {
    int step = 18;
    for (int i = 0; i < step; ++i)
    {
      Point2d p;
      p <<
        w/2. + 100*cos(i*2*M_PI/step),
        h/2. + 150*sin(i*2*M_PI/step);
      poly.push_back(p);

      p.array() += 90;
      clip.push_back(p);
    }
  }
  draw_poly(poly, Red8);
  draw_poly(clip, Blue8);
  get_key();

  int numIter = 1000;
  timer.restart();
  for (int i = 0; i < numIter; ++i)
    res = sutherland_hodgman(poly, clip);
  elapsed = timer.elapsedMs()/numIter;
  cout << "Intersection computation time = " << elapsed << " milliseconds"
       << endl;

  draw_poly(res, Green8,5);
  get_key();

  return 0;
}