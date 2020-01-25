// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Core/ArrayIterators/Iterator Classes"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/MultiArray.hpp>

#include <iostream>


using namespace DO::Sara;
using namespace std;


struct TestFixtureFor2DIterators
{
protected:
  using Image = MultiArray<Vector2i, 2, RowMajor>;
  Image image;

public:
  TestFixtureFor2DIterators()
  {
    image.resize(5, 10);
    for (auto i = 0; i < image.rows(); ++i)
      for (auto j = 0; j < image.cols(); ++j)
        image(i, j) = Vector2i{i, j};
  }
};

BOOST_FIXTURE_TEST_SUITE(Test2DIterators, TestFixtureFor2DIterators)

BOOST_AUTO_TEST_CASE(test_row_major_prefix_increment)
{
  auto it = image.begin_array();
  for (auto i = 0; i < image.rows(); ++i)
  {
    for (auto j = 0; j < image.cols(); ++j)
    {
      BOOST_REQUIRE_EQUAL(*it, Vector2i(i, j));
      BOOST_REQUIRE_EQUAL(it.position(), Vector2i(i, j));
      BOOST_REQUIRE_EQUAL(it.operator->(),
                          image.data() + jump(Vector2i(i, j), image.strides()));
      ++it;
    }
  }
  BOOST_CHECK(it.end());
}

BOOST_AUTO_TEST_CASE(test_row_major_postfix_increment)
{
  auto it = image.begin_array();
  for (auto i = 0; i < image.rows(); ++i)
    for (auto j = 0; j < image.cols(); ++j)
      BOOST_REQUIRE_EQUAL(*(it++), Vector2i(i, j));
}

BOOST_AUTO_TEST_CASE(test_row_major_prefix_decrement)
{
  auto it = image.begin_array();
  it += image.sizes() - Vector2i::Ones();

  for (auto i = image.rows() - 1; i >= 0; --i)
  {
    for (auto j = image.cols() - 1; j >= 0; --j)
    {
      BOOST_REQUIRE_EQUAL(*it, Vector2i(i, j));
      BOOST_REQUIRE_EQUAL(it.position(), Vector2i(i, j));
      BOOST_REQUIRE_EQUAL(it.operator->(),
                          image.data() + jump(Vector2i(i, j), image.strides()));
      --it;
    }
  }
  BOOST_CHECK(it.end());
}

BOOST_AUTO_TEST_CASE(test_row_major_postfix_decrement)
{
  auto it = image.begin_array();
  it += image.sizes() - Vector2i::Ones();

  for (auto i = image.rows() - 1; i >= 0; --i)
    for (auto j = image.cols() - 1; j >= 0; --j)
      BOOST_REQUIRE_EQUAL(*(it--), Vector2i(i, j));
}

BOOST_AUTO_TEST_CASE(test_special_dereferencing_operator)
{
  auto it = image.begin_array();
  BOOST_REQUIRE_EQUAL(it(Vector2i(1, 2)), Vector2i(1, 2));
  BOOST_REQUIRE_EQUAL(it(1, 2), Vector2i(1, 2));
}

BOOST_AUTO_TEST_CASE(test_const_multiarray_iterator)
{
  const auto& const_image = image;
  auto it = const_image.begin_array();
  while (!it.end())
    ++it;
}

BOOST_AUTO_TEST_CASE(test_equality_and_inequality_comparisons)
{
  const auto& const_image = image;

  // Equalities.
  BOOST_CHECK(image.begin_array() == const_image.begin_array());
  BOOST_CHECK(image.begin_array() == image.begin_array()++);
  BOOST_CHECK(image.begin_array() == image.begin());
  BOOST_CHECK(image.begin_array() == image.data());

  BOOST_REQUIRE(*image.begin_array() == *const_image.begin_array());

  // Inequalities.
  BOOST_CHECK(image.begin_array() != ++image.begin_array());
  BOOST_CHECK(image.begin_array() != image.data() + 1);

  BOOST_CHECK(image.begin_array() != ++const_image.begin_array());
  BOOST_CHECK(image.begin_array() != const_image.begin() + 1);
  BOOST_CHECK(image.begin_array() != const_image.data() + 1);
}

BOOST_AUTO_TEST_SUITE_END()


class TestFixtureFor2DSubarrayIterators : public TestFixtureFor2DIterators
{
protected:
  Vector2i start;
  Vector2i end;

public:
  TestFixtureFor2DSubarrayIterators()
    : TestFixtureFor2DIterators()
  {
    start << 1, 2;
    end << 4, 8;
  }
};

BOOST_FIXTURE_TEST_SUITE(Test2DSubarrayIterators,
                         TestFixtureFor2DSubarrayIterators)

BOOST_AUTO_TEST_CASE(test_row_major_prefix_increment)
{
  auto it = image.begin_subarray(start, end);
  for (auto i = start(0); i < end(0); ++i)
  {
    for (auto j = start(1); j < end(1); ++j)
    {
      BOOST_REQUIRE_EQUAL(*it, Vector2i(i, j));
      BOOST_REQUIRE_EQUAL(it.position(), Vector2i(i, j));
      BOOST_REQUIRE_EQUAL(it.operator->(),
                          image.data() + jump(Vector2i(i, j), image.strides()));
      ++it;
    }
  }
  BOOST_CHECK(it.end());
}

BOOST_AUTO_TEST_CASE(test_row_major_postfix_increment)
{
  auto it = image.begin_subarray(start, end);
  for (auto i = start(0); i < end(0); ++i)
    for (auto j = start(1); j < end(1); ++j)
      BOOST_REQUIRE_EQUAL(*(it++), Vector2i(i, j));
}

BOOST_AUTO_TEST_CASE(test_row_major_prefix_decrement)
{
  auto it = image.begin_subarray(start, end);
  it += end - start - Vector2i::Ones();

  for (auto i = end(0) - 1; i >= start(0); --i)
  {
    for (auto j = end(1) - 1; j >= start(1); --j)
    {
      BOOST_REQUIRE_EQUAL(*it, Vector2i(i, j));
      BOOST_REQUIRE_EQUAL(it.position(), Vector2i(i, j));
      BOOST_REQUIRE_EQUAL(it.operator->(),
                          image.data() + jump(Vector2i(i, j), image.strides()));
      --it;
    }
  }
  BOOST_CHECK(it.end());
}

BOOST_AUTO_TEST_CASE(test_row_major_postfix_decrement)
{
  auto it = image.begin_subarray(start, end);
  it += end - start - Vector2i::Ones();

  for (auto i = end(0) - 1; i >= start(0); --i)
    for (auto j = end(1) - 1; j >= start(1); --j)
      BOOST_REQUIRE_EQUAL(*(it--), Vector2i(i, j));
}

BOOST_AUTO_TEST_CASE(test_special_dereferencing_operator)
{
  auto it = image.begin_subarray(start, end);

  auto relative_position = Vector2i{2, -1};
  auto absolute_position = Vector2i{start + relative_position};

  BOOST_REQUIRE_EQUAL(it(relative_position), absolute_position);
  BOOST_REQUIRE_EQUAL(it(relative_position), absolute_position);
}

BOOST_AUTO_TEST_CASE(test_const_multiarray_iterator)
{
  const Image& const_image = image;
  auto it = const_image.begin_subarray(start, end);
  while (!it.end())
    ++it;
}

BOOST_AUTO_TEST_CASE(test_equality_and_inequality_comparisons)
{
  const Image& const_image = image;

  // Equalities.
  BOOST_CHECK(image.begin_subarray(start, end) ==
              const_image.begin_subarray(start, end));
  BOOST_CHECK(image.begin_subarray(Vector2i::Zero(), image.sizes()) ==
              image.begin_array());
  BOOST_CHECK(image.begin_subarray(Vector2i::Zero(), image.sizes()) ==
              image.begin());
  BOOST_CHECK(image.begin_subarray(Vector2i::Zero(), image.sizes()) ==
              image.data());

  BOOST_REQUIRE(*image.begin_subarray(Vector2i::Zero(), image.sizes()) ==
                *const_image.begin_array());

  // Inequalities.
  BOOST_CHECK(image.begin_subarray(Vector2i::Zero(), image.sizes()) !=
              ++image.begin_array());
  BOOST_CHECK(image.begin_subarray(Vector2i::Zero(), image.sizes()) !=
              image.data() + 1);

  BOOST_CHECK(image.begin_subarray(Vector2i::Zero(), image.sizes()) !=
              ++const_image.begin_array());
  BOOST_CHECK(image.begin_subarray(Vector2i::Zero(), image.sizes()) !=
              const_image.begin() + 1);
  BOOST_CHECK(image.begin_subarray(Vector2i::Zero(), image.sizes()) !=
              const_image.data() + 1);
}

BOOST_AUTO_TEST_SUITE_END()

class TestFixtureFor2DSteppedSubarrayIterators : public TestFixtureFor2DIterators
{
protected:
  Vector2i start;
  Vector2i end;
  Vector2i steps;

public:
  TestFixtureFor2DSteppedSubarrayIterators()
  {
    start << 2, 3;
    end << 6, 11;
    steps << 2, 3;

    image.resize(20, 20);
    for (auto i = 0; i < image.rows(); ++i)
      for (auto j = 0; j < image.cols(); ++j)
        image(i, j) = Vector2i{i, j};
  }
};

BOOST_FIXTURE_TEST_SUITE(Test2DSteppedSubarrayIterators,
                         TestFixtureFor2DSteppedSubarrayIterators)

BOOST_AUTO_TEST_CASE(test_row_major_prefix_increment)
{
  const auto true_visited_coords =
      std::vector<Vector2i>{Vector2i{2, 3}, Vector2i{2, 6}, Vector2i{2, 9},
                            Vector2i{4, 3}, Vector2i{4, 6}, Vector2i{4, 9}};
  const auto true_visited_values = true_visited_coords;

  auto true_visited_ptrs = std::vector<Vector2i *>{};
  for (const auto& c : true_visited_coords)
    true_visited_ptrs.push_back(image.data() + jump(c, image.strides()));

  auto visited_values = std::vector<Vector2i>{};
  auto visited_ptrs = std::vector<Vector2i *>{};
  auto visited_coords = std::vector<Vector2i>{};

  auto it = image.begin_stepped_subarray(start, end, steps);
  for (; !it.end(); ++it)
  {
    visited_values.push_back(*it);
    visited_coords.push_back(it.position());
    visited_ptrs.push_back(it.operator->());
  }

  BOOST_CHECK(true_visited_coords == visited_coords);
  BOOST_CHECK(true_visited_values == visited_values);
  BOOST_CHECK(true_visited_ptrs == visited_ptrs);
  BOOST_CHECK(it.end());
}

BOOST_AUTO_TEST_CASE(test_row_major_postfix_decrement)
{
  const auto true_visited_coords =
      std::vector<Vector2i>{Vector2i{4, 9}, Vector2i{4, 6}, Vector2i{4, 3},
                            Vector2i{2, 9}, Vector2i{2, 6}, Vector2i{2, 3}};
  const auto true_visited_values = true_visited_coords;

  auto true_visited_ptrs = std::vector<Vector2i *>{};
  for (const auto& c : true_visited_coords)
    true_visited_ptrs.push_back(image.data() + jump(c, image.strides()));

  auto visited_values = std::vector<Vector2i>{};
  auto visited_ptrs = std::vector<Vector2i *>{};
  auto visited_coords = std::vector<Vector2i>{};

  auto it = image.begin_stepped_subarray(start, end, steps);
  it += Vector2i{1, 2};
  for (; !it.end(); --it)
  {
    visited_values.push_back(*it);
    visited_coords.push_back(it.position());
    visited_ptrs.push_back(it.operator->());
  }

  BOOST_CHECK(true_visited_coords == visited_coords);
  BOOST_CHECK(true_visited_values == visited_values);
  BOOST_CHECK(true_visited_ptrs == visited_ptrs);
  BOOST_CHECK(it.end());
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_FIXTURE_TEST_SUITE(Test2DAxisterators, TestFixtureFor2DIterators)

BOOST_AUTO_TEST_CASE(test_equality_and_inequality_comparisons)
{
  const Image& const_image = image;
  auto it = image.begin_array();

  // Check equalities.
  BOOST_CHECK(it.x() == image.begin_array());
  BOOST_CHECK(it.x() == image.begin_array()++);
  BOOST_CHECK(it.x() == image.begin());
  BOOST_CHECK(it.x() == image.data());
  BOOST_CHECK(it.x() == it.y());

  BOOST_CHECK(it.x() == const_image.begin_array());
  BOOST_CHECK(it.x() == const_image.begin_array()++);
  BOOST_CHECK(it.x() == const_image.begin());
  BOOST_CHECK(it.x() == const_image.data());
  BOOST_CHECK(it.x() == const_image.begin_array().y());

  BOOST_REQUIRE_EQUAL(*it.x(), *it.y());

  // Inequalities.
  BOOST_CHECK(it.x() != ++image.begin_array());
  BOOST_CHECK(it.x() != image.begin() + 1);
  BOOST_CHECK(it.x() != image.data() + 1);

  BOOST_CHECK(it.x() != ++const_image.begin_array());
  BOOST_CHECK(it.x() != const_image.begin() + 1);
  BOOST_CHECK(it.x() != const_image.data() + 1);
}

BOOST_AUTO_TEST_CASE(test_iterations)
{
  auto it = image.begin_array();
  BOOST_CHECK(++it.x() == image.begin() + image.stride(0));
  BOOST_CHECK(--it.x() == image.begin());
}

BOOST_AUTO_TEST_CASE(test_arithmetic_operations)
{
  auto it = image.begin_array();

  // Addition.
  it.x() += 2;
  BOOST_CHECK(it == image.begin() + image.stride(0) * 2);

  // Subtraction.
  it.x() -= 2;
  BOOST_CHECK(it == image.begin());

  // Out of range.
  BOOST_CHECK_THROW(it.x() -= 1, std::out_of_range);
  BOOST_CHECK_THROW(it.x() += image.rows(), std::out_of_range);
}

BOOST_AUTO_TEST_SUITE_END()


class TestFixtureFor3DIterators
{
protected:
  using Volume = MultiArray<Vector3i, 3, RowMajor>;
  Volume volume;

public:
  TestFixtureFor3DIterators()
  {
    volume.resize(2, 5, 7);
    for (auto i = 0; i < volume.rows(); ++i)
      for (auto j = 0; j < volume.cols(); ++j)
        for (auto k = 0; k < volume.depth(); ++k)
          volume(i, j, k) = Vector3i{i, j, k};
  }
};

BOOST_FIXTURE_TEST_SUITE(Test3DIterators, TestFixtureFor3DIterators)

BOOST_AUTO_TEST_CASE(test_row_major_increment)
{
  auto it = volume.begin_array();
  for (auto i = 0; i < volume.rows(); ++i)
  {
    for (auto j = 0; j < volume.cols(); ++j)
    {
      for (auto k = 0; k < volume.depth(); ++k)
      {
        BOOST_REQUIRE_EQUAL(*it, Vector3i(i, j, k));
        BOOST_REQUIRE_EQUAL(it.position(), Vector3i(i, j, k));
        BOOST_REQUIRE_EQUAL(it.operator->(),
                            volume.data() +
                                jump(Vector3i(i, j, k), volume.strides()));
        ++it;
      }
    }
  }
  BOOST_CHECK(it.end());
}

BOOST_AUTO_TEST_CASE(test_row_major_decrement)
{
  auto it = volume.begin_array();
  it += volume.sizes() - Vector3i::Ones();

  for (auto i = volume.rows() - 1; i >= 0; --i)
  {
    for (auto j = volume.cols() - 1; j >= 0; --j)
    {
      for (auto k = volume.depth() - 1; k >= 0; --k)
      {
        BOOST_REQUIRE_EQUAL(*it, Vector3i(i, j, k));
        BOOST_REQUIRE_EQUAL(it.position(), Vector3i(i, j, k));
        BOOST_REQUIRE_EQUAL(it.operator->(),
                            volume.data() +
                                jump(Vector3i(i, j, k), volume.strides()));
        --it;
      }
    }
  }
  BOOST_CHECK(it.end());
}

BOOST_AUTO_TEST_CASE(test_special_dereferencing_operator)
{
  auto it = volume.begin_array();
  BOOST_REQUIRE_EQUAL(it(Vector3i(1, 2, 4)), Vector3i(1, 2, 4));
  BOOST_REQUIRE_EQUAL(it(1, 2, 4), Vector3i(1, 2, 4));
}

BOOST_AUTO_TEST_CASE(test_delta)
{
  auto it = volume.begin_array();
  const int X = 0;
  const int Y = 1;
  int dx = 1;
  int dy = 1;

  BOOST_REQUIRE_EQUAL(it.delta(X, dx, Y, dy), Vector3i(0 + dx, 0 + dy, 0));

  auto value = it.delta<X, Y>(dx, dy);
  BOOST_REQUIRE_EQUAL(value, Vector3i(0 + dx, 0 + dy, 0));
}

BOOST_AUTO_TEST_SUITE_END()
