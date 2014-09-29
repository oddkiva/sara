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

  for (size_t i = 0; i != signal.size(); ++i)
  {
    if (i > signal.size()-3)
      EXPECT_EQ(1, signal[i]);
    else
      EXPECT_EQ(3, signal[i]);
  }
}


TEST(TestLinearFitering, test_apply_row_based_filter)
{
  Image<float> image(3, 3);
  image.matrix() << 1, 2, 3,
                    1, 2, 3,
                    1, 2, 3;
  const float kernel[] = { -1./2, 0, 1./2 };

  apply_row_based_filter(image, image, kernel, 3);

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
