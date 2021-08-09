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

#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Graphics.hpp>


using namespace std;
using namespace DO::Sara;


GRAPHICS_MAIN()
{
  Image<Rgb8> I;
  cout << src_path("../../../data/ksmall.jpg") << endl;
  if (!load(I, src_path("../../../data/ksmall.jpg")))
  {
    cerr << "Error: could not open 'ksmall.jpg' file" << endl;
    return 1;
  }
  int w = I.width(), h = I.height();
  int x = 0, y = 0;

  create_window(2 * w, h);

  Timer drawTimer;
  drawTimer.restart();
  double elapsed;
  for (int i = 0; i < 1; ++i)
  {
    clear_window();
    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        draw_point(x, y, I(x, y));
        draw_point(w + x, y, I(x, y));
#ifdef Q_OS_MAC
        microsleep(10);
#endif
      }
    }
  }

  elapsed = drawTimer.elapsed();
  std::cout << "Drawing time: " << elapsed << "s" << std::endl;

  click();

  int step = 2;
  Timer t;
  clear_window();
  while (true)
  {
    microsleep(10);
    display(I, {x, y});
    clear_window();

    x += step;
    if (x < 0 || x > w)
      step *= -1;

    if (t.elapsed() > 2)
      break;
  }
  close_window(active_window());

  cout << "Finished!" << endl;

  return 0;
}
