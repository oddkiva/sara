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

#include <DO/Sara/Graphics.hpp>

using namespace std;
using namespace DO::Sara;

GRAPHICS_MAIN()
{
  create_window(300, 300);
  get_key();

  // We can get the active window with the following function.
  Window w1 = active_window();
  Window w2 = create_window(100, 100);
  set_active_window(w2);
  get_key();
  close_window(w2);

  set_active_window(w1);
  draw_circle(120, 120, 30, Red8);
  get_key();
  close_window();

  return 0;
}
