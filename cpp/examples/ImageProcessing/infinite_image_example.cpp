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
  auto image = Image<Rgb8>{};
  imread(image, "/home/david/GitHub/DO-CV/sara/data/sunflowerField.jpg");

  // Extend the image in an infinite domain with a mirror periodic padding.
  auto pad = ConstantPadding<Rgb8>(Black8);
  //auto pad = PeriodicPadding();
  auto inf_image = make_infinite(image, pad);

  const auto border = Vector2i::Ones() * 50;

  const Vector2i begin = -border;
  const Vector2i end = image.sizes() + border;

  //const auto repeat = 2;
  //const Vector2i begin = -repeat * image.sizes();
  //const Vector2i end = repeat * image.sizes();

  auto ext_image = Image<Rgb8>{end - begin};

  Timer t;
  double start, finish;
  const auto num_iter = 10;

  t.restart();
  start = t.elapsed_ms();

  for (int i = 0; i < num_iter; ++i)
  {
    auto src_c = inf_image.begin_subarray(begin, end);
    auto dst_i = ext_image.begin_array();
    for (; !dst_i.end(); ++src_c, ++dst_i)
      //std::cout << src_c.coords().transpose() << std::endl;
      *dst_i = *src_c;
  }

  finish = t.elapsed_ms();
  std::cout << (finish - start) / num_iter << " ms" << std::endl;

  create_window(ext_image.sizes());
  display(ext_image);
  get_key();

  close_window();

  return EXIT_SUCCESS;
}
