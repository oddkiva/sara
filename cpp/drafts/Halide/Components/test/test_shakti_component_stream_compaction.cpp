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

#include <drafts/Halide/Components/Evaluation.hpp>
#include <drafts/Halide/Components/TinyLinearAlgebra.hpp>


using DO::Shakti::HalideBackend::eval;
using DO::Shakti::HalideBackend::Matrix2;
using DO::Shakti::HalideBackend::Matrix3;

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
  Func in_flattened;
  in_flattened(x) = in(x % w, (x / w) % h, (x / (w * h)) % c, x / (w * h * c));

  Func prefix_sum;
  prefix_sum(x) = cast<int32_t>(0);

  RDom range(1, size - 1);
  prefix_sum(range) = select(in_flattened(range - 1) != 0,  //
                             prefix_sum(range - 1) + 1,     //
                             prefix_sum(range - 1));

  RDom in_range(0, size);
  Func compacted_indices;
  compacted_indices(x) = -1;
  compacted_indices(clamp(prefix_sum(in_range), 0, size - 1)) =
      select(in_flattened(in_range) != 0, in_range, -1);

  auto num_non_zeros_fn = Func{};
  num_non_zeros_fn() = cast<int32_t>(0);
  num_non_zeros_fn() += select(in_flattened(in_range) != 0, 1, 0);

  auto non_zeros_x = Func{};
  auto non_zeros_y = Func{};
  auto non_zeros_c = Func{};
  auto non_zeros_n = Func{};

  const auto flat_index = compacted_indices(x);
  non_zeros_x(x) = select(flat_index != -1, flat_index % w, -1);
  non_zeros_y(x) = select(flat_index != -1, (flat_index / w) % h, -1);
  non_zeros_c(x) = select(flat_index != -1, (flat_index / (w * h)) % c, -1);
  non_zeros_n(x) = select(flat_index != -1, flat_index / (w * h * c), -1);

  Buffer<int32_t> sum = prefix_sum.realize(8);
  Buffer<int32_t> indices = compacted_indices.realize(size);
  Buffer<int32_t> num_non_zeros = num_non_zeros_fn.realize();
  Buffer<int32_t> xs = non_zeros_x.realize(size);
  Buffer<int32_t> ys = non_zeros_y.realize(size);
  Buffer<int32_t> cs = non_zeros_c.realize(size);
  Buffer<int32_t> ns = non_zeros_n.realize(size);

  SARA_CHECK(num_non_zeros());
  print_1d(sum);

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
