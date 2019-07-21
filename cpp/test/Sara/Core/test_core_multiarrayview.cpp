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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>
#include <DO/Sara/Core/MultiArray/MultiArray.hpp>
#include <DO/Sara/Core/Numpy.hpp>
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

BOOST_AUTO_TEST_CASE(test_multiarrayview_flatten)
{
  auto r = range(10).reshape(Vector2i{2, 5});

  auto r_flatten = r.flatten();

  BOOST_CHECK(r_flatten == range(10));
}


struct AxisSlice
{
  int start{0};
  int stop{0};
  int step{1};
};


template <typename T, int N, int O>
struct ViewSliced
{
  using vector_type = Matrix<int, N, 1>;
  using view_type = MultiArrayView<T, N, O>;

  auto operator()(const vector_type& x) -> T&
  {
    if ((x.array() < 0).any() || ((slice_sizes - x).array() <= 0).any())
      throw std::runtime_error{
          "Coordinates are not in the valid slices range!"};
    const vector_type y = start + x.dot(steps);
    return view(y);
  }

  auto operator()(const Matrix<int, N, 1>& x) const -> const T&
  {
    if ((x.array() < 0).any() || ((slice_sizes - x).array() <= 0).any())
      throw std::runtime_error{
          "Coordinates are not in the valid slices range!"};
    const vector_type y = start + x.dot(steps);
    return view(y);
  }

  auto begin() const
  {
    return view.begin_stepped_subarray(start, stop, steps);
  }

  auto end() const
  {
    return view.end_stepped_subarray(start, stop, steps);
  }

  auto make_copy() const
  {
    auto view_copy = MultiArray<T, N, O>{slice_sizes};
    std::transform(std::begin(*this), std::end(*this), std::begin(view_copy),
                   [](const auto& v) { return v; });
    return view_copy;
  }

  view_type view;
  vector_type start;
  vector_type stop;
  vector_type steps;
  vector_type slice_sizes;
};


template <typename T, int N, int O>
auto slice(const MultiArrayView<T, N, O>& x,
           const std::vector<AxisSlice>& slices)
{
  const auto ixs = range(N);
  auto start = Matrix<int, N, 1>{};
  auto stop = Matrix<int, N, 1>{};
  auto step = Matrix<int, N, 1>{};
  std::for_each(std::begin(ixs), std::end(ixs), [&](int i) {
    start[i] = slices[i].start;
    stop[i] = slices[i].stop;
    step[i] = slices[i].step;
  });

  const auto slice_sizes =
      x.begin_stepped_subarray(start, stop, step).stepped_subarray_sizes();

  return ViewSliced<T, N, O>{x, start, stop, step, slice_sizes};
}

BOOST_AUTO_TEST_CASE(test_slice_view)
{
  const auto X = range(24).cast<float>().reshape(Vector2i{6, 4});
  auto X_sliced = slice(X, {{1, 6, 2}, {1, 4, 2}}).make_copy();
  SARA_DEBUG << "X_sliced =\n" << X_sliced.matrix() << std::endl;
}


BOOST_AUTO_TEST_SUITE_END()
