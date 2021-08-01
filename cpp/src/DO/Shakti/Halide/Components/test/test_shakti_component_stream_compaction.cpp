// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Halide Backend/Stream Compaction"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <DO/Shakti/Halide/Components/StreamCompaction.hpp>


namespace halide = DO::Shakti::HalideBackend;

using namespace Halide;


static void print_1d(const Buffer<int32_t>& result)
{
  std::cout << "{ ";
  const char* prefix = "";
  for (int i = 0; i < result.dim(0).extent(); i++)
  {
    std::cout << prefix << result(i);
    prefix = ", ";
  }
  std::cout << "}\n";
}


BOOST_AUTO_TEST_CASE(test_stream_compaction)
{
  int8_t vals[] = {0, 1, 0, 0, -1, 0, 1, -1};
  Buffer<int8_t> in(vals, 2, 2, 2, 1);
  SARA_CHECK(in.width());
  SARA_CHECK(in.height());
  SARA_CHECK(in.channels());
  SARA_CHECK(in.dim(3).extent());

  const auto w = in.dim(0).extent();
  const auto h = in.dim(1).extent();
  const auto c = in.dim(2).extent();
  const auto n = in.dim(3).extent();
  const auto size = w * h * c * n;

  Var x;
  auto in_flattened = halide::flatten_4d(in, x, w, h, c);

  auto compacted_indices = halide::non_zeros_indices(in_flattened, x, size);

  auto [non_zeros_x, non_zeros_y, non_zeros_c, non_zeros_n] =
      halide::unflatten_4d(compacted_indices, x, w, h, c);

  Buffer<int32_t> indices = compacted_indices.realize({size});
  Buffer<int32_t> xs = non_zeros_x.realize({size});
  Buffer<int32_t> ys = non_zeros_y.realize({size});
  Buffer<int32_t> cs = non_zeros_c.realize({size});
  Buffer<int32_t> ns = non_zeros_n.realize({size});

  std::cout << "Flat indices =" << std::endl;
  print_1d(indices);

  std::cout << "x =" << std::endl;
  print_1d(xs);
  std::cout << "y =" << std::endl;
  print_1d(ys);
  std::cout << "c =" << std::endl;
  print_1d(cs);
  std::cout << "n =" << std::endl;
  print_1d(ns);
}
