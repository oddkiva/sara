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
  load(image, src_path("../../../data/ksmall.jpg"));

  create_graphics_view(image.width(), image.height());

  for (int i = 0; i < 10; ++i)
  {
    auto pixmap = add_pixmap(image);
    if (pixmap == nullptr)
      cerr << "Error image display" << endl;
  }

  while (get_key() != KEY_ESCAPE);
  close_window();

  return 0;
}
