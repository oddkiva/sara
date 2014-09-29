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


#include <gtest/gtest.h>

#include <DO/Defines.hpp>
#include <DO/Core/DebugUtilities.hpp>
#include <DO/ImageProcessing/LinearFiltering.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO;


TEST(TestLinearFiltering, test_convolve_array)
{
  std::vector<float> signal(10, 1);
  std::vector<float> kernel(3, 1);

  convolve_array(&signal[0], &kernel[0], signal.size()-2, kernel.size());

  //for (int i = 0; i < 2; ++i)
  //  ASSERT_EQ(signal[i], 1);
  for (size_t i = 0; i != signal.size(); ++i)
  {
    CHECK(i);
    if (i > signal.size()-3)
      EXPECT_EQ(1, signal[i]);
    else
      EXPECT_EQ(3, signal[i]);
  }
}


int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}