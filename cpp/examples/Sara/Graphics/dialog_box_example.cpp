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
  Image<Rgb8> image;
  if (!load_from_dialog_box(image))
    return EXIT_FAILURE;

  create_window(image.width(), image.height(), "Image loaded from dialog box");
  display(image);
  get_key();

  close_window();

  return EXIT_SUCCESS;
}
