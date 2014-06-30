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
#include <DO/Core/Locator.hpp>
#include <DO/Core/DebugUtilities.hpp>
#include "MultiArrayTestingFunctions.hpp"

using namespace DO;
using namespace std;

template <int StorageOrder>
void testOffsetComputation()
{
  // Create coords and dims.
  const int coords[] = { 2, 3, 4 };
  const int dims[] = { 10, 20, 30 };
  // Check offset computations.
  if (StorageOrder == RowMajor)
  {
    EXPECT_EQ((Offset<1, StorageOrder>::eval(coords, dims)), 2);
    EXPECT_EQ((Offset<2, StorageOrder>::eval(coords, dims)), 2*20+3);
    EXPECT_EQ((Offset<3, StorageOrder>::eval(coords, dims)), 2*20*30+3*30+4);
  }
  else
  {
    EXPECT_EQ((Offset<1, StorageOrder>::eval(coords, dims)), 2);
    EXPECT_EQ((Offset<2, StorageOrder>::eval(coords, dims)), 3*10+2);
    EXPECT_EQ((Offset<3, StorageOrder>::eval(coords, dims)), 4*10*20+3*10+2);
  }
}

template <int StorageOrder>
void testStrideComputation()
{
  // Create dims.
  const int dims[] = { 10, 20, 30 };
  // Check stride computations.
  int strides[3];
  if (StorageOrder == ColMajor)
  {
    // Column major strides
    Offset<3, StorageOrder>::eval_strides(strides, dims);
    EXPECT_EQ(strides[0], 1);
    EXPECT_EQ(strides[1], 10);
    EXPECT_EQ(strides[2], 200);
  }
  else
  {
    // Row major strides
    Offset<3, StorageOrder>::eval_strides(strides, dims);
    EXPECT_EQ(strides[0], 600);
    EXPECT_EQ(strides[1], 30);
    EXPECT_EQ(strides[2], 1);
  }
}

template <int StorageOrder>
void testRangeIterator()
{
  typedef MultiArray<Color4f, 3, StorageOrder> Volume;
  Volume volume;

  // Check MultiArray class.
  initVolume(volume);

  // Check Locator class.
  checkLocatorIncrement(volume);

  // Decrement.
  checkLocatorDecrement(volume);

  // Pot-pourri testing.
  checkLocatorPotPourri(volume);
};

template <int StorageOrder>
void testSubrangeIterator()
{
  typedef MultiArray<Color3f, 3, StorageOrder> Volume;
  typedef typename Volume::range_iterator range_iterator;
  typedef typename Volume::subrange_iterator subrange_iterator;

  // Data
  Vector3i dims(5, 10, 15);
  Volume vol(dims);

  // Work variable
  Vector3i coords( Vector3i::Zero() );

  range_iterator it(vol.begin_range());
  for ( ; it != vol.end_range(); ++it)
  {
    // 'r_it.coords()' is denoted as $c_i$.
    // 'dims[i]' is denoted as $d_i$.
    //
    // Check that $0 \leq c_i < d_i$.
    *it = it.coords().template cast<float>();
    //r_it.check();

    // 1. Check that $\min_i c_i \geq 0$.
    ASSERT_GE( (it->template cast<int>().array().minCoeff()), 0 );
    // 2. Check that $\max_i d_i - c_i \geq 1$.
    ASSERT_GE( (dims - it->template cast<int>()).array().maxCoeff(), 1 );
  }

  Vector3i start(1,1,1), end(3, 3, 3);

  subrange_iterator it2(vol.begin_subrange(start, end));
  for ( ; it2 != vol.end_subrange(); ++it2)
  {
    // 'sr_it.coords()' is denoted as $c_i$.
    // 'start[i]' is denoted as $a_i$.
    // 'end[i]' is denoted as $b_i$.
    //
    // Check that $a_i \leq c_i < b_i$.
    *it2 = it2.coords().template cast<float>();
    //sr_it.check();

    // 1. Check that $\min_i c_i - a_i \geq 0$.
    ASSERT_GE( (it2->template cast<int>() - start).array().minCoeff(), 0 );
    // 2. Check that $\max_i b_i - c_i \geq 1$.
    ASSERT_GE( (end - it2->template cast<int>()).array().maxCoeff(), 1 );
  }
};

TEST(DO_Core_Test, MultiArrayIndexComputation)
{
  // Row-major based tests.
  testOffsetComputation<RowMajor>();  
  testStrideComputation<RowMajor>();
  // Column-major based tests.
  testOffsetComputation<ColMajor>();
  testStrideComputation<ColMajor>();
}

TEST(DO_Core_Test, NDIterator)
{
  testRangeIterator<RowMajor>();
  testRangeIterator<ColMajor>();
  testSubrangeIterator<RowMajor>();
  testSubrangeIterator<ColMajor>();
}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}