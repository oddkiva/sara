// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Core/MultiArray/MultiArrayView Class"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>
#include <DO/Sara/Core/MultiArray/MultiArray.hpp>
#include <DO/Sara/Core/Tensor.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestMultiArrayView)

BOOST_AUTO_TEST_CASE(test_multiarrayview_empty)
{
  {
    auto array_view = MultiArrayView<int, 2>{};
    BOOST_CHECK(array_view.empty());
  }

  {
    auto r = std::vector<int>(3 * 4);
    auto array_view = MultiArrayView<int, 2>{r.data(), {3, 4}};
    BOOST_CHECK(!array_view.empty());
  }
}

BOOST_AUTO_TEST_CASE(test_multiarrayview_1d)
{
  auto r = std::vector<int>(10);
  std::iota(std::begin(r), std::end(r), 0);

  const auto const_r_view = MultiArrayView<int, 1>{r.data(), int(r.size())};
  for (auto i = 0; i < int(const_r_view.size()); ++i)
    BOOST_CHECK_EQUAL(const_r_view(i), i);

  auto r_view = MultiArrayView<int, 1>{r.data(), int(r.size())};
  for (auto i = 0u; i < const_r_view.size(); ++i)
    r_view(i) = 0;

  for (auto i = 0u; i < const_r_view.size(); ++i)
    BOOST_CHECK_EQUAL(r_view(i), 0);
}

BOOST_AUTO_TEST_CASE(test_multiarrayview_cast)
{
  auto r = std::vector<int>(10);
  std::iota(std::begin(r), std::end(r), 0);

  const auto const_r_view = tensor_view(r);
  for (auto i = 0; i < int(const_r_view.size()); ++i)
    BOOST_CHECK_EQUAL(const_r_view(i), i);

  auto rd = const_r_view.cast<double>();

  auto true_rd = Tensor_<double, 1>(10);
  std::iota(std::begin(true_rd), std::end(true_rd), 0);

  BOOST_CHECK(true_rd.vector() == rd.vector());
}

BOOST_AUTO_TEST_SUITE_END()
