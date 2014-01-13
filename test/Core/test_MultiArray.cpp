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

#include "CoreTesting.hpp"
#include "MultiArrayTestingFunctions.hpp"

template <typename T, int N, int StorageOrder>
void testMultiArrayDefaultConstructor()
{
  typedef MultiArray<T, N, StorageOrder> NDArray;
  NDArray ndarray;
  ASSERT_EQ(ndarray.data(), ndarray.begin());
  ASSERT_EQ(ndarray.begin(), ndarray.end());
  ASSERT_EQ(ndarray.sizes(), NDArray::coords_type::Zero());
}

template <typename T, int N, int StorageOrder>
MultiArray<T, N, StorageOrder> foo()
{ return MultiArray<T, N, StorageOrder>(MultiArray<T, N, StorageOrder>::coords_type::Ones()); }

template <typename T, int N, int StorageOrder>
void testMultiArrayCopyConstructor()
{
  typedef MultiArray<T, N, StorageOrder> NDArray;
  typedef typename NDArray::vector_type Coords;
  Coords sizes(Coords::Ones()*5);
  for (int i = 1; i < sizes.size(); ++i)
    sizes(i) = sizes(i-1)*2;

  NDArray array1(sizes);
  array1.array().fill(1);

  NDArray array2(array1);
  array2.array() *= T(2);

  for (int i = 0; i < array1.array().size(); ++i)
  {
    ASSERT_EQ(array1.array()(i), 1);
    ASSERT_EQ(array2.array()(i), 2);
  }

  array1 = foo<T,N,StorageOrder>();
}

template <typename T, int N, int StorageOrder>
void testMultiArrayAll()
{
  testMultiArrayDefaultConstructor<T, N, StorageOrder>();
  testMultiArrayCopyConstructor<T, N, StorageOrder>();
}

template <class ChannelType>
class MultiArrayTest : public testing::Test
{
protected:
  typedef testing::Test Base;
  MultiArrayTest() : Base() {}
};

typedef testing::Types<
  /*uchar, ushort, uint, char, short, int, float, double,
  Vector2i, Vector2f, Vector2d, Vector2cf, Vector2cd,
  Vector3i, Vector3f, Vector3d, Vector3cf, Vector3cd,
  Vector4i, Vector4f, Vector4d, Vector4cf, Vector4cd*/
  int
> ArrayElementTypes;

TYPED_TEST_CASE_P(MultiArrayTest);

TYPED_TEST_P(MultiArrayTest, multiArrayTest)
{
  typedef TypeParam T;

  testMultiArrayAll<T, 2, RowMajor>();
}

REGISTER_TYPED_TEST_CASE_P(MultiArrayTest, multiArrayTest);

INSTANTIATE_TYPED_TEST_CASE_P(DO_Core_Test, MultiArrayTest, ArrayElementTypes);

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}