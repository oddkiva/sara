// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/MultiArray/InfiniteMultiArrayView.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>


using namespace std;
using namespace DO::Sara;


GRAPHICS_MAIN()
{
  auto image = Image<float>{};
  imread(image, "/home/david/GitHub/DO-CV/sara/data/sunflowerField.jpg");

  auto kernel = Image<float>{10, 10};
  kernel.matrix() = MatrixXf::Ones(10, 10) / 100.f;

  create_window(image.sizes());
  display(image);
  get_key();
  close_window();

  return EXIT_SUCCESS;
}
