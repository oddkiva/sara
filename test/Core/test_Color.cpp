// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include "gtest/gtest.h"
#include <DO/Core.hpp>
#include <iostream>
#include <list>
#include <utility>

using namespace DO;
using namespace std;

template <class ChannelType>
class RgbTest : public testing::Test
{
protected:
  typedef testing::Test Base;
  RgbTest() : Base() {}
};

typedef testing::Types<
  unsigned char, unsigned short, unsigned int,
  char, short, int, float, double
> ChannelTypes;

TYPED_TEST_CASE_P(RgbTest);

TYPED_TEST_P(RgbTest, assignmentTest)
{
  typedef TypeParam ChannelType, T;
  typedef Color<ChannelType, Rgb> Color3;

  Color3 a1(black<ChannelType>());
  EXPECT_EQ(a1, black<ChannelType>());

  Color3 a2(1, 2, 3);
  EXPECT_EQ(a2(0), 1);
  EXPECT_EQ(a2(1), 2);
  EXPECT_EQ(a2(2), 3);

  // Test mutable getters.
  a1.template channel<R>() = 64;
  a1.template channel<G>() = 12;
  a1.template channel<B>() = 124;
  EXPECT_EQ(a1.template channel<R>(), static_cast<T>(64));
  EXPECT_EQ(a1.template channel<G>(), static_cast<T>(12));
  EXPECT_EQ(a1.template channel<B>(), static_cast<T>(124));

  // Test immutable getters.
  const Color3& ca1 = a1;
  EXPECT_EQ(ca1.template channel<R>(), static_cast<T>(64));
  EXPECT_EQ(ca1.template channel<G>(), static_cast<T>(12));
  EXPECT_EQ(ca1.template channel<B>(), static_cast<T>(124));

  // Test assignment for each channel of the RGB pixel.
  red(a1) = 89; green(a1) = 50; blue(a1) = 12;
  EXPECT_EQ(red(a1), static_cast<T>(89));
  EXPECT_EQ(green(a1), static_cast<T>(50));
  EXPECT_EQ(blue(a1), static_cast<T>(12));

  // 
  EXPECT_EQ(red(ca1), static_cast<T>(89));
  EXPECT_EQ(green(ca1), static_cast<T>(50));
  EXPECT_EQ(blue(ca1), static_cast<T>(12));
}

REGISTER_TYPED_TEST_CASE_P(RgbTest, assignmentTest);
INSTANTIATE_TYPED_TEST_CASE_P(DO_Core_Test, RgbTest, ChannelTypes);

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}
