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
  cout << "Advanced event handling" << endl;
  Window W = create_window(1024, 768);
  set_active_window(W);
  Image<Color3ub> I;
  load(I, src_path("../../datasets/ksmall.jpg"));

  Event ev;
  do
  {
    get_event(500, ev);  // Wait an event (return if no event for 500ms)
    switch (ev.type)
    {
    case NO_EVENT:
      break;
    case MOUSE_PRESSED_AND_MOVED:
      clear_window();
      display(I, ev.mousePos - I.sizes() * 3 / 8, 0.75);
      cout << "Mouse moved. Position = " << endl << ev.mousePos << endl;
      break;
    case KEY_PRESSED:
      cout << "Key " << ev.key << " pressed" << endl;
      break;
    case KEY_RELEASED:
      cout << "Key " << ev.key << " released" << endl;
      break;
    case MOUSE_PRESSED:
      clear_window();
      display(I, ev.mousePos - I.sizes() * 3 / 8, 0.75);
      cout << "Button " << ev.buttons << " pressed" << endl;
      break;
    case MOUSE_RELEASED:
      cout << "Button " << ev.buttons << " released" << endl;
      break;
    }
  } while (ev.type != KEY_PRESSED || ev.key != KEY_UP);
  close_window(W);

  return 0;
}
