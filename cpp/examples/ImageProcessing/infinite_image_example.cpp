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


template <typename T, int N, int O, typename Padding>
void stepped_safe_crop(MultiArrayView<T, N, O>& dst,
                       const MultiArrayView<T, N, O>& src,
                       const Matrix<int, N, 1>& begin,
                       const Matrix<int, N, 1>& end,
                       const Matrix<int, N, 1>& steps,
                       const Padding& padding)
{
  const auto inf_src = make_infinite(src, padding);
  auto src_i = inf_src.begin_stepped_subarray(begin, end, steps);

  const auto sizes = src_i.stepped_subarray_sizes();
  if (dst.sizes() != sizes)
  {
    std::ostringstream oss;
    oss << "Error: destination sizes " << dst.sizes().transpose()
        << "is invalid and must be: " << sizes.transpose();
    throw std::domain_error{oss.str()};
  }

  for (auto dst_i = dst.begin(); dst_i != dst.end(); ++src_i, ++dst_i)
    *dst_i = *src_i;
}


GRAPHICS_MAIN()
{
  auto image = Image<Rgb8>{};
  imread(image, "/home/david/GitHub/DO-CV/sara/data/sunflowerField.jpg");

  // Extend the image in an infinite domain with a mirror periodic padding.
  auto pad = ConstantPadding<Rgb8>(Black8);
  //auto pad = PeriodicPadding();
  auto inf_image = make_infinite(image, pad);

  const auto border = Vector2i::Ones() * 50;

#ifdef STEPPED
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
      *dst_i = *src_c;
  }

  finish = t.elapsed_ms();
  std::cout << (finish - start) / num_iter << " ms" << std::endl;
#else
  const Vector2i begin = {0, 0};
  const Vector2i end = image.sizes();
  const Vector2i steps = {3, 3};

  auto sizes = Vector2i{};
  for (int i = 0; i < 2; ++i)
  {
    const auto modulo = (end[i] - begin[i]) % steps[i];
    sizes[i] = (end[i] - begin[i]) / steps[i] + int(modulo != 0);
  }
  auto ext_image = Image<Rgb8>{sizes};

  Timer t;
  double start, finish;
  const auto num_iter = 1;

  t.restart();
  start = t.elapsed_ms();

  for (int i = 0; i < num_iter; ++i)
  {
    stepped_safe_crop(ext_image, image, begin, end, steps, pad);
  }

  finish = t.elapsed_ms();
  std::cout << (finish - start) / num_iter << " ms" << std::endl;
#endif

  create_window(ext_image.sizes());
  display(ext_image);
  get_key();

  close_window();

  return EXIT_SUCCESS;
}
