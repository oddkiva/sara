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

#include <gtest/gtest.h>

#include <DO/Sara/Match.hpp>


using namespace std;
using namespace DO::Sara;


TEST(TestMatch, test_make_index_match)
{
  auto m = make_index_match(0, 1000);
  EXPECT_EQ(m.index_x(), 0);
  EXPECT_EQ(m.index_y(), 1000);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}