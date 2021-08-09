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
  cout << "Basic mouse functions" << endl;

  auto w = create_window(512, 512, "Mouse");
  set_antialiasing();
  draw_text(10, 20, "Please click anywhere", White8, 18, 1, false, true, false, 2);

  click();

  draw_text(10, 50, "click again (left=BLUE, middle=RED, right=done)", White8, 12);

  int button;
  Point2i p;
  while ((button = get_mouse(p)) != MOUSE_RIGHT_BUTTON)
  {
    Rgb8 color;
    if (button == MOUSE_LEFT_BUTTON)
      color = Blue8;
    else if (button == MOUSE_MIDDLE_BUTTON)
      color = Red8;
    else
      color = Black8;
    fill_circle(p, 5, color);
  }

  close_window(w);

  return 0;
}
