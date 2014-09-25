// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <stdint.h>

#include <gtest/gtest.h>

#include <DO/Core/Pixel/Typedefs.hpp>


using namespace std;
using namespace DO;


TEST(TestPixelTypedefs, test_typedefs)
{
  EXPECT_EQ(Color3ub(255, 0, 0), Red8);
  EXPECT_EQ(Color3ub(0, 255, 0), Green8);
  EXPECT_EQ(Color3ub(0, 0, 255), Blue8);
}


// ========================================================================== //
int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}