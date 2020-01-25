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
#include <DO/Sara/ImageProcessing.hpp>

using namespace std;
using namespace DO::Sara;


GRAPHICS_MAIN()
{
  Image<Rgb8> image;
  if (!load_from_dialog_box(image))
    return EXIT_FAILURE;

  // Resize image.
  create_window((image.sizes() * 2).eval(), "Image loaded from dialog box");
  display(enlarge(image, 2));
  display(image);
  display(reduce(image, 2));
  get_key();

  // Pixelwise operations.
  auto res = image.cwise_transform([](const Rgb8& color) {
    Rgb64f color_64f;
    smart_convert_color(color, color_64f);
    color_64f =
    color_64f.cwiseProduct(color_64f);
    return color_64f;
  });

  display(res);
  get_key();

  close_window();

  return EXIT_SUCCESS;
}
