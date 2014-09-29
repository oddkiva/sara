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

  convolve_array(&signal[0], &kernel[0],
                 static_cast<int>(signal.size())-2,
                 static_cast<int>(kernel.size()));

  for (size_t i = 0; i != signal.size(); ++i)
  {
    if (i > signal.size()-3)
      EXPECT_EQ(1, signal[i]);
    else
      EXPECT_EQ(3, signal[i]);
  }
}


class TestFilters : public testing::Test
{
protected:
  Image<float> _src_image;
  vector<float> _kernel;

  TestFilters() : testing::Test()
  {
    _src_image.resize(3, 3);
    _src_image.matrix() <<
      1, 2, 3,
      1, 2, 3,
      1, 2, 3;

    _kernel.resize(3);
    _kernel[0] = -1./2;
    _kernel[1] =  0;
    _kernel[2] =  1./2;
  }
};


  MatrixXf true_matrix(3, 3);
  true_matrix << 0.5, 1, 0.5,
                 0.5, 1, 0.5,
                 0.5, 1, 0.5;

  EXPECT_MATRIX_EQ(true_matrix, image.matrix());
}


int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}
