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

#include <DO/Sara/ImageProcessing/Extrema.hpp>


using namespace std;
using namespace DO::Sara;


TEST(TestLocalExtremum, test_local_extremum)
{
  // Simple test case.
  Image<float> I(10,10);
  I.matrix().fill(1.f);

  // Local maximality and minimality
  CompareWithNeighborhood3<greater_equal, float> greater_equal33;
  CompareWithNeighborhood3<less_equal, float> less_equal33;
  // Strict local maximality and minimality
  CompareWithNeighborhood3<greater, float> greater33;
  CompareWithNeighborhood3<less, float> less33;

  // Check local maximality
  EXPECT_FALSE(greater33(I(1,1), 1, 1, I, true));
  EXPECT_FALSE(greater33(I(1,1), 1, 1, I, false));
  EXPECT_TRUE(greater_equal33(I(1,1),1,1,I,true));
  EXPECT_TRUE(greater_equal33(I(1,1),1,1,I,false));
  // Check local minimality
  EXPECT_FALSE(less33(I(1,1), 1, 1, I, true));
  EXPECT_FALSE(less33(I(1,1), 1, 1, I, false));
  EXPECT_TRUE(less_equal33(I(1,1),1,1,I,true));
  EXPECT_TRUE(less_equal33(I(1,1),1,1,I,false));
  // Check that aliases are correctly defined.
  EXPECT_FALSE(StrictLocalMax<float>()(1, 1, I));
  EXPECT_FALSE(StrictLocalMin<float>()(1, 1, I));
  vector<Point2i> maxima;
  vector<Point2i> minima;

  maxima = strict_local_maxima(I);
  EXPECT_TRUE(maxima.empty());
  maxima = local_maxima(I);
  EXPECT_TRUE(maxima.size() == 8*8);

  minima = strict_local_minima(I);
  EXPECT_TRUE(minima.empty());
  minima = local_minima(I);
  EXPECT_TRUE(minima.size() == 8*8);

  I(1,1) = 10.f;
  I(7,7) = 10.f;
  EXPECT_TRUE(greater33(I(1,1), 1, 1, I, false));
  EXPECT_FALSE(greater33(I(1,1), 1, 1, I, true));
  EXPECT_TRUE(LocalMax<float>()(1, 1, I));
  EXPECT_TRUE(StrictLocalMax<float>()(1, 1, I));
  EXPECT_FALSE(LocalMin<float>()(1, 1, I));
  EXPECT_FALSE(StrictLocalMin<float>()(1, 1, I));

  maxima = strict_local_maxima(I);
  EXPECT_EQ(maxima.size(), 2u);
  minima = strict_local_minima(I);
  EXPECT_TRUE(minima.empty());

  I.matrix() *= -1;
  EXPECT_TRUE(less33(I(1,1), 1, 1, I, false));
  EXPECT_FALSE(less33(I(1,1), 1, 1, I, true));
  EXPECT_TRUE(StrictLocalMin<float>()(1, 1, I));
  EXPECT_TRUE(LocalMin<float>()(1, 1, I));
  maxima = strict_local_maxima(I);
  minima = strict_local_minima(I);

  EXPECT_TRUE(maxima.empty());
  EXPECT_EQ(minima.size(), 2u);
}


TEST(TestLocalExtremum, test_local_scale_space_extremum)
{
  ImagePyramid<double> I;
  I.reset(1,3,1.6f,pow(2., 1./3.));
  for (int i = 0; i < 3; ++i)
  {
    I(i,0).resize(10,10);
    I(i,0).matrix().fill(1);
  }
  EXPECT_FALSE(StrictLocalScaleSpaceMax<double>()(1,1,1,0,I));
  EXPECT_FALSE(StrictLocalScaleSpaceMin<double>()(1,1,1,0,I));

  // Local scale-space extrema test 1
  I(1,1,1,0) = 10.f;
  I(7,7,1,0) = 10.f;
  EXPECT_TRUE(StrictLocalScaleSpaceMax<double>()(1,1,1,0,I));
  EXPECT_FALSE(StrictLocalScaleSpaceMin<double>()(1,1,1,0,I));

  vector<Point2i> maxima, minima;
  maxima = strict_local_scale_space_maxima(I,1,0);
  minima = strict_local_scale_space_minima(I,1,0);
  EXPECT_EQ(maxima.size(), 2u);
  EXPECT_TRUE(minima.empty());

  // Local scale-space extrema test 2
  I(1,1,1,0) *= -1.f;
  I(7,7,1,0) *= -1.f;
  maxima = strict_local_scale_space_maxima(I,1,0);
  minima = strict_local_scale_space_minima(I,1,0);
  EXPECT_FALSE(LocalScaleSpaceMax<double>()(1,1,1,0,I));
  EXPECT_FALSE(StrictLocalScaleSpaceMax<double>()(1,1,1,0,I));
  EXPECT_TRUE(LocalScaleSpaceMin<double>()(1,1,1,0,I));
  EXPECT_TRUE(StrictLocalScaleSpaceMin<double>()(1,1,1,0,I));
  EXPECT_TRUE(maxima.empty());
  EXPECT_EQ(minima.size(), 2u);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}