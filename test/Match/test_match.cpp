// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <limits>

#include <gtest/gtest.h>

#include <DO/Sara/Match.hpp>


using namespace std;
using namespace DO::Sara;

TEST(TestMatch, test_default_constructor)
{
  auto m = Match{};
  EXPECT_TRUE(m.x_pointer() == nullptr);
  EXPECT_TRUE(m.y_pointer() == nullptr);
  EXPECT_THROW(m.x(), runtime_error);
  EXPECT_THROW(m.y(), runtime_error);
  EXPECT_EQ(m.x_index(), -1);
  EXPECT_EQ(m.y_index(), -1);
  EXPECT_EQ(m.score(), numeric_limits<float>::max());
  EXPECT_EQ(m.rank(), -1);
  EXPECT_EQ(m.matching_direction(), Match::SourceToTarget);
}

TEST(TestMatch, test_make_index_match)
{
  auto m = make_index_match(0, 1000);
  EXPECT_EQ(m.x_index(), 0);
  EXPECT_EQ(m.y_index(), 1000);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}