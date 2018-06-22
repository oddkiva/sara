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

#define BOOST_TEST_MODULE "Core/ArrayIterators/Basic Functions"

#include <DO/Sara/Core/ArrayIterators.hpp>

#include <boost/test/unit_test.hpp>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_SUITE(TestStrideComputer)

BOOST_AUTO_TEST_CASE(test_row_major_strides_computation_2d)
{
  auto sizes = Vector2i{10, 20};
  auto strides = Vector2i{20, 1};

  BOOST_REQUIRE_EQUAL(StrideComputer<RowMajor>::eval(sizes), strides);
}

BOOST_AUTO_TEST_CASE(test_col_major_strides_computation_2d)
{
  auto sizes = Vector2i{10, 20};
  auto strides = Vector2i{1, 10};

  BOOST_REQUIRE_EQUAL(StrideComputer<ColMajor>::eval(sizes), strides);
}

BOOST_AUTO_TEST_CASE(test_row_major_stride_computation_3d)
{
  auto sizes = Vector3i{10, 20, 30};
  auto strides = Vector3i{20 * 30, 30, 1};

  BOOST_REQUIRE_EQUAL(StrideComputer<RowMajor>::eval(sizes), strides);
}

BOOST_AUTO_TEST_CASE(test_col_major_stride_computation_3d)
{
  auto sizes = Vector3i{10, 20, 30};
  auto strides = Vector3i{1, 10, 10 * 20};

  BOOST_REQUIRE_EQUAL(StrideComputer<ColMajor>::eval(sizes), strides);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(TestJump)

BOOST_AUTO_TEST_CASE(test_jump_2d)
{
  auto coords = Vector2i{2, 3};
  auto sizes = Vector2i{10, 20};
  auto strides = StrideComputer<RowMajor>::eval(sizes);

  BOOST_REQUIRE_EQUAL(2 * 20 + 3, jump(coords, strides));
}

BOOST_AUTO_TEST_CASE(test_jump_3d)
{
  auto coords = Vector3i{2, 3, 4};
  auto sizes = Vector3i{10, 20, 30};
  auto strides = StrideComputer<RowMajor>::eval(sizes);

  BOOST_REQUIRE_EQUAL(jump(coords, strides), 2 * 20 * 30 + 3 * 30 + 4);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(TestPositionIncrementer)

BOOST_AUTO_TEST_CASE(test_row_major_incrementer_2d)
{
  auto stop = false;
  auto start = Vector2i{2, 3};
  auto end = Vector2i{5, 10};

  auto coords = start;
  for (auto i = start(0); i < end(0); ++i)
  {
    for (auto j = start(1); j < end(1); ++j)
    {
      BOOST_REQUIRE(!stop);
      BOOST_REQUIRE_EQUAL(coords, Vector2i(i, j));
      PositionIncrementer<RowMajor>::apply(coords, stop, start, end);
    }
  }
  BOOST_REQUIRE(stop);
}

BOOST_AUTO_TEST_CASE(test_row_major_stepped_incrementer_2d)
{
  auto stop = false;
  auto start = Vector2i{2, 3};
  auto end = Vector2i{5, 10};
  auto steps = Vector2i{2, 3};

  auto coords = start;

  auto true_visited_coords = std::vector<Vector2i>{
    Vector2i{2, 3}, Vector2i{2, 6}, Vector2i{2, 9},
    Vector2i{4, 3}, Vector2i{4, 6}, Vector2i{4, 9},
  };

  auto visited_coords = std::vector<Vector2i>{};
  for (auto i = 0u; i < true_visited_coords.size(); ++i)
  {
    visited_coords.push_back(coords);
    BOOST_REQUIRE(!stop);
    PositionIncrementer<RowMajor>::apply(coords, stop, start, end, steps);
  }
  BOOST_REQUIRE(stop);

  BOOST_REQUIRE(true_visited_coords == visited_coords);
}

BOOST_AUTO_TEST_CASE(test_col_major_incrementer_2d)
{
  auto stop = false;
  auto start = Vector2i{2, 3};
  auto end = Vector2i{5, 10};

  auto coords = start;
  for (auto j = start(1); j < end(1); ++j)
  {
    for (auto i = start(0); i < end(0); ++i)
    {
      BOOST_REQUIRE(!stop);
      BOOST_REQUIRE_EQUAL(coords, Vector2i(i, j));
      PositionIncrementer<ColMajor>::apply(coords, stop, start, end);
    }
  }
  BOOST_REQUIRE(stop);
}

BOOST_AUTO_TEST_CASE(test_col_major_stepped_incrementer_2d)
{
  auto stop = false;
  auto start = Vector2i{2, 3};
  auto end = Vector2i{5, 10};
  auto steps = Vector2i{2, 3};

  auto coords = start;

  auto true_visited_coords = std::vector<Vector2i>{
      Vector2i{2, 3}, Vector2i{4, 3},  // Col 0
      Vector2i{2, 6}, Vector2i{4, 6},  // Col 1
      Vector2i{2, 9}, Vector2i{4, 9},  // Col 2
  };

  auto visited_coords = std::vector<Vector2i>{};
  for (auto i = 0u; i < true_visited_coords.size(); ++i)
  {
    visited_coords.push_back(coords);
    BOOST_REQUIRE(!stop);
    PositionIncrementer<ColMajor>::apply(coords, stop, start, end, steps);
  }
  BOOST_REQUIRE(stop);

  BOOST_REQUIRE(true_visited_coords == visited_coords);
}


BOOST_AUTO_TEST_CASE(test_row_major_decrementer_2d)
{
  auto stop = false;
  auto start = Vector2i{2, 3};
  auto end = Vector2i{5, 10};

  auto coords = Vector2i{};
  coords.array() = end.array() - 1;
  for (auto i = end(0) - 1; i >= start(0); --i)
  {
    for (auto j = end(1) - 1; j >= start(1); --j)
    {
      BOOST_REQUIRE(!stop);
      BOOST_REQUIRE_EQUAL(coords, Vector2i(i, j));
      PositionDecrementer<RowMajor>::apply(coords, stop, start, end);
    }
  }
  BOOST_REQUIRE(stop);
}

BOOST_AUTO_TEST_CASE(test_row_major_stepped_decrementer_2d)
{
  auto stop = false;
  auto start = Vector2i{2, 3};
  auto end = Vector2i{5, 10};
  auto steps = Vector2i{2, 3};

  auto coords = Vector2i{4, 9};

  auto true_visited_coords = std::vector<Vector2i>{
    Vector2i{4, 9}, Vector2i{4, 6}, Vector2i{4, 3},
    Vector2i{2, 9}, Vector2i{2, 6}, Vector2i{2, 3},
  };

  auto visited_coords = std::vector<Vector2i>{};
  for (auto i = 0u; i < true_visited_coords.size(); ++i)
  {
    visited_coords.push_back(coords);
    BOOST_REQUIRE(!stop);
    PositionDecrementer<RowMajor>::apply(coords, stop, start, end, steps);
  }
  BOOST_REQUIRE(stop);

  BOOST_REQUIRE(true_visited_coords == visited_coords);
}

BOOST_AUTO_TEST_CASE(test_col_major_decrementer_2d)
{
  bool stop = false;
  auto start = Vector2i{2, 3};
  auto end = Vector2i{5, 10};

  auto coords = Vector2i{};
  coords.array() = end.array() - 1;
  for (int j = end(1) - 1; j >= start(1); --j)
  {
    for (int i = end(0) - 1; i >= start(0); --i)
    {
      BOOST_REQUIRE(!stop);
      BOOST_REQUIRE_EQUAL(coords, Vector2i(i, j));
      PositionDecrementer<ColMajor>::apply(coords, stop, start, end);
    }
  }
  BOOST_REQUIRE(stop);
}

BOOST_AUTO_TEST_CASE(test_col_major_stepped_decrementer_2d)
{
  auto stop = false;
  auto start = Vector2i{2, 3};
  auto end = Vector2i{5, 10};
  auto steps = Vector2i{2, 3};

  auto coords = Vector2i{4, 9};

  auto true_visited_coords = std::vector<Vector2i>{
      Vector2i{4, 9}, Vector2i{2, 9}, // Col 2
      Vector2i{4, 6}, Vector2i{2, 6}, // Col 1
      Vector2i{4, 3}, Vector2i{2, 3}, // Col 0
  };

  auto visited_coords = std::vector<Vector2i>{};
  for (auto i = 0u; i < true_visited_coords.size(); ++i)
  {
    visited_coords.push_back(coords);
    BOOST_REQUIRE(!stop);
    PositionDecrementer<ColMajor>::apply(coords, stop, start, end, steps);
  }
  BOOST_REQUIRE(stop);

  BOOST_REQUIRE(true_visited_coords == visited_coords);
}


BOOST_AUTO_TEST_SUITE_END()
