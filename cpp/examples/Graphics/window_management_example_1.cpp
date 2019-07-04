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
  // Open a 300x200 window.
  Window window = create_window(300, 200, "A window");

  // A 150x100 filled RED rectangle with top-left corner at (20, 10).
  fill_rect(20, 10, 150, 100, Red8);

  // Wait for a click.
  click();

  // Close window.
  close_window(window);

  return 0;
}
