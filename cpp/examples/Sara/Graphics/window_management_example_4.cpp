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
  // Open 6 windows aligned in 2x3 grid.
  vector<Window> windows;
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      windows.push_back(create_window(200, 200,
                                      "Window #" + to_string(i * 3 + j),
                                      300 * j + 300, 300 * i + 50));
      set_active_window(windows.back());
      fill_rect(0, 0, get_width(windows.back()), get_height(windows.back()),
                Color3ub(rand() % 255, rand() % 255, rand() % 255));
      draw_text(100, 100, to_string(i * 3 + j), Yellow8, 15);
      cout << "Pressed '" << char(any_get_key()) << "'" << endl;
    }
  }

  // Make the last window active.
  set_active_window(windows.back());
  // Click on any windows to continue.
  any_click();

  // Close the 6 windows in LIFO order.
  for (size_t i = 0; i < windows.size(); ++i)
  {
    any_get_key();
    cout << "Closing window #" << i << endl;
    close_window(windows[i]);
  }

  return 0;
}
