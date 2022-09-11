// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE                                                      \
  "Geometry/Algorithms/Suzuki-Abe Border Following Algorithm"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Geometry/Algorithms/BorderFollowing.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestSuzukiAbeBorderFollowingAlgorithm)

BOOST_AUTO_TEST_CASE(test_on_square)
{
  auto pic = Image<std::uint8_t>{10, 10};
  // clang-format off
  pic.matrix() <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // clang-format on

  auto border = Image<int>{pic.sizes()};
  border.matrix() = pic.matrix().cast<int>();

  const auto p = Eigen::Vector2i{2, 2};
  const auto p2 = Eigen::Vector2i{2, 1};

  auto curve = std::vector<Eigen::Vector2i>{};
  auto nbd = 2;

  border(p) = 2;
  follow_border(border, curve, p, p2, nbd);

  const auto true_curve = std::vector<Eigen::Vector2i>{
      {2, 2},  //
      {2, 3},  //
      {2, 4},  //
      {2, 5},  //
      {2, 6},  //
      {2, 7},  //
      {3, 7},  //
      {4, 7},  //
      {5, 7},  //
      {6, 7},  //
      {7, 7},  //
      {7, 6},  //
      {7, 5},  //
      {7, 4},  //
      {7, 3},  //
      {7, 2},  //
      {6, 2},  //
      {5, 2},  //
      {4, 2},  //
      {3, 2}   //
  };
  BOOST_CHECK_EQUAL_COLLECTIONS(curve.begin(), curve.end(), true_curve.begin(),
                                true_curve.end());
}

BOOST_AUTO_TEST_CASE(test_on_shape_1)
{
  auto pic = Image<std::uint8_t>{10, 10};
  // clang-format off
  pic.matrix() <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // clang-format on

  auto border = Image<int>{pic.sizes()};
  border.matrix() = pic.matrix().cast<int>();

  auto curve = std::vector<Eigen::Vector2i>{};
  const auto p = Eigen::Vector2i{4, 1};
  const auto p2 = Eigen::Vector2i{3, 1};
  auto nbd = 2;

  border(p) = 2;
  follow_border(border, curve, p, p2, nbd);

  std::cout << border.matrix() << std::endl << std::endl;

  const auto true_curve = std::vector<Eigen::Vector2i>{
      {4, 1},  //
      {3, 2},  //
      {2, 3},  //
      {2, 4},  //
      {3, 5},  //
      {4, 6},  //
      {5, 7},  //
      {6, 7},  //
      {7, 7},  //
      {7, 6},  //
      {7, 5},  //
      {7, 4},  //
      {7, 3},  //
      {7, 2},  //
      {6, 2},  //
      {5, 1}   //
  };
  BOOST_CHECK_EQUAL_COLLECTIONS(curve.begin(), curve.end(), true_curve.begin(),
                                true_curve.end());
}

BOOST_AUTO_TEST_CASE(test_on_shape_2)
{
  auto pic = Image<std::uint8_t>{10, 10};
  // clang-format off
  pic.matrix() <<
    0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // clang-format on

  auto border = Image<int>{pic.sizes()};
  border.matrix() = pic.matrix().cast<int>();

  auto curve = std::vector<Eigen::Vector2i>{};
  const auto p = Eigen::Vector2i{1, 0};
  const auto p2 = Eigen::Vector2i{0, 0};
  auto nbd = 2;

  border(p) = 2;
  follow_border(border, curve, p, p2, nbd);

  std::cout << border.matrix() << std::endl << std::endl;

  const auto true_curve = std::vector<Eigen::Vector2i>{
      {1, 0},  //
      {1, 1},  //
      {2, 2},  //
      {2, 3},  //
      {3, 3},  //
      {4, 3},  //
      {5, 3},  //
      {5, 2},  //
      {4, 2},  //
      {3, 1},  //
      {2, 0}   //
  };
  BOOST_CHECK_EQUAL_COLLECTIONS(curve.begin(), curve.end(), true_curve.begin(),
                                true_curve.end());
}

BOOST_AUTO_TEST_CASE(test_on_shape_3)
{
  auto pic = Image<std::uint8_t>{10, 10};
  // clang-format off
  pic.matrix() <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // clang-format on

  const auto borders = suzuki_abe_follow_border(pic);
  BOOST_CHECK_EQUAL(borders.size(), 3u);
  for (const auto& border : borders)
  {
    SARA_CHECK(border.second.id);
    SARA_CHECK(border.second.parent);
    SARA_CHECK(static_cast<int>(border.second.type));
    SARA_CHECK(border.second.curve.size());
    for (const auto& p : border.second.curve)
      std::cout << p.transpose() << std::endl;
    std::cout << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(test_on_shape_4)
{
  auto pic = Image<std::uint8_t>{10, 10};
  // clang-format off
  pic.matrix() <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // clang-format on

  const auto borders = suzuki_abe_follow_border(pic);
  BOOST_CHECK_EQUAL(borders.size(), 3u);
  for (const auto& border : borders)
  {
    SARA_CHECK(border.second.id);
    SARA_CHECK(border.second.parent);
    SARA_CHECK(static_cast<int>(border.second.type));
    SARA_CHECK(border.second.curve.size());
    for (const auto& p : border.second.curve)
      std::cout << p.transpose() << std::endl;
    std::cout << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(test_on_edge)
{
  auto pic = Image<std::uint8_t>{10, 10};
  // clang-format off
  pic.matrix() <<
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 1, 0, 0, 1, 1, 1, 1, 0,
    0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  // clang-format on

  auto border = Image<int>{pic.sizes()};
  border.matrix() = pic.matrix().cast<int>();

  const auto p = Eigen::Vector2i{2, 1};
  const auto p2 = Eigen::Vector2i{1, 1};

  auto curve = std::vector<Eigen::Vector2i>{};
  auto nbd = 2;

  border(p) = 2;
  follow_border(border, curve, p, p2, nbd);

  std::cout << border.matrix() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
