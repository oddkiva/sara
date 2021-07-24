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
  Window w = create_window(300, 300);
  set_active_window(w);

  Event e;
  do
  {
    get_event(1, e);
    fill_rect(rand() % 300, rand() % 300, rand() % 50, rand() % 50,
              Rgb8(rand() % 256, rand() % 256, rand() % 256));
    // microSleep(100);  // sometimes if you don't put this, the program
                         // freezes in some machines.
  } while (e.key != KEY_ESCAPE);

  cout << "Finished!" << endl;

  close_window(active_window());

  return 0;
}
