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

#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>
#include <DO/Core/Timer.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{
  int w, h;
  w = h = 400;

  Timer timer;
  double elapsed;

  openWindow(w,h);
  setAntialiasing();

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
  drawPoly(poly, Red8);
  drawPoly(clip, Blue8);
  getKey();

  int numIter = 1000;
  timer.restart();
  for (int i = 0; i < numIter; ++i)
    res = sutherlandHodgman(poly, clip);
  elapsed = timer.elapsedMs()/numIter;
  cout << "Intersection computation time = " << elapsed << " milliseconds" << endl;
 
  drawPoly(res, Green8,5);
  getKey();

  return 0;
}