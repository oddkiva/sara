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

#include <vector>

#include <gtest/gtest.h>

#include <DO/Core/MultiArray.hpp>

#include "../AssertHelpers.hpp"


using namespace DO;
using namespace std;


// =============================================================================
// Tests on generic constructors.
template <typename MultiArrayType>
class Test_multiarray_constructors : public testing::Test {};
TYPED_TEST_CASE_P(Test_multiarray_constructors);

TYPED_TEST_P(Test_multiarray_constructors,
             test_default_constructor_and_base_getters)
{
  typedef TypeParam MultiArray;
  MultiArray multi_array;
  EXPECT_FALSE(multi_array.data());
  EXPECT_FALSE(multi_array.begin());
  EXPECT_FALSE(multi_array.end());
  EXPECT_MATRIX_EQ(multi_array.sizes(),
                   MultiArray::vector_type::Zero().eval());
  EXPECT_MATRIX_EQ(multi_array.strides(),
                   MultiArray::vector_type::Zero().eval());
}

TYPED_TEST_P(Test_multiarray_constructors,
             test_constructor_with_sizes_and_base_getters)
{
  typedef TypeParam MultiArray;
  typedef typename MultiArray::vector_type Vector;
  typedef StrideComputer<MultiArray::StorageOrder> StrideComputer;

  // Create sizes, one for each dimension.
  Vector sizes;
  for (int i = 0; i < sizes.size(); ++i)
    sizes(i) = (i+1)*10;
  // Compute the strides.
  Vector strides = StrideComputer::eval(sizes);
  // Compute the raw size of the multi-array.
  size_t raw_size = static_cast<size_t>( accumulate(
    sizes.data(), sizes.data() + sizes.size(), 1,
    multiplies<int>()) );
  // Create the array with the right sizes.
  MultiArray multi_array(sizes);

  EXPECT_EQ(multi_array.sizes(), sizes);
  EXPECT_EQ(multi_array.strides(), strides);
  EXPECT_EQ(multi_array.size(), raw_size);

  // Check size values in each dimension.
  for (int i = 0; i < sizes.size(); ++i)
    ASSERT_EQ(multi_array.size(i), sizes(i));
  // Check stride values in each dimension.
  for (int i = 0; i < sizes.size(); ++i)
    ASSERT_EQ(multi_array.stride(i), strides(i));
}

REGISTER_TYPED_TEST_CASE_P(Test_multiarray_constructors,
                           test_default_constructor_and_base_getters,
                           test_constructor_with_sizes_and_base_getters);

typedef testing::Types<
  MultiArray<unsigned char, 2, ColMajor>,
  MultiArray<unsigned char, 2, RowMajor>,
  MultiArray<unsigned char, 3, ColMajor>,
  MultiArray<unsigned char, 3, RowMajor>,
  MultiArray<unsigned char, 4, ColMajor>,
  MultiArray<unsigned char, 4, RowMajor>,
  MultiArray<unsigned char, 5, ColMajor>,
  MultiArray<unsigned char, 5, RowMajor>
> Test_multiarray_constructors_types;
INSTANTIATE_TYPED_TEST_CASE_P(TestMultiArray,
                              Test_multiarray_constructors,
                              Test_multiarray_constructors_types);


// =============================================================================
// Tests on convenience constructor (2D case)
template <typename MultiArrayType>
class Test_multiarray_convenience_constructor_2d_and_getters
  : public testing::Test {};
TYPED_TEST_CASE_P(Test_multiarray_convenience_constructor_2d_and_getters);

TYPED_TEST_P(Test_multiarray_convenience_constructor_2d_and_getters, run)
{
  typedef TypeParam MultiArray;
  typedef StrideComputer<MultiArray::StorageOrder> StrideComputer;

  MultiArray multi_array(10, 20);

  EXPECT_EQ(multi_array.sizes(), Vector2i(10, 20));
  EXPECT_EQ(multi_array.rows(), 10);
  EXPECT_EQ(multi_array.cols(), 20);
  EXPECT_EQ(multi_array.strides(), StrideComputer::eval(Vector2i(10, 20)));
  EXPECT_EQ(multi_array.size(), 10u*20u);
}
REGISTER_TYPED_TEST_CASE_P(
  Test_multiarray_convenience_constructor_2d_and_getters, run);

typedef testing::Types<
  MultiArray<int, 2, ColMajor>,
  MultiArray<int, 2, RowMajor>,
  MultiArray<float, 2, ColMajor>,
  MultiArray<float, 2, RowMajor>
> Test_multiarray_convenience_constructor_2d_types;
INSTANTIATE_TYPED_TEST_CASE_P(
    TestMultiArray,
    Test_multiarray_convenience_constructor_2d_and_getters,
    Test_multiarray_convenience_constructor_2d_types);


// =============================================================================
// Tests on convenience constructor (3D case)
template <typename MultiArrayType>
class Test_multiarray_convenience_constructor_3d_and_getters
  : public testing::Test {};
TYPED_TEST_CASE_P(Test_multiarray_convenience_constructor_3d_and_getters);

TYPED_TEST_P(Test_multiarray_convenience_constructor_3d_and_getters, run)
{
  typedef TypeParam MultiArray;
  typedef StrideComputer<MultiArray::StorageOrder> StrideComputer;

  MultiArray multi_array(10, 20, 50);

  EXPECT_EQ(multi_array.sizes(), Vector3i(10, 20, 50));
  EXPECT_EQ(multi_array.rows(), 10);
  EXPECT_EQ(multi_array.cols(), 20);
  EXPECT_EQ(multi_array.depth(), 50);
  EXPECT_EQ(multi_array.strides(), StrideComputer::eval(Vector3i(10, 20, 50)));
  EXPECT_EQ(multi_array.size(), 10u*20u*50u);
}
REGISTER_TYPED_TEST_CASE_P(
  Test_multiarray_convenience_constructor_3d_and_getters, run);

typedef testing::Types<
  MultiArray<int, 3, ColMajor>,
  MultiArray<int, 3, RowMajor>,
  MultiArray<float, 3, ColMajor>,
  MultiArray<float, 3, RowMajor>
> Test_multiarray_convenience_constructor_3d_types;
INSTANTIATE_TYPED_TEST_CASE_P(
    TestMultiArray,
    Test_multiarray_convenience_constructor_3d_and_getters,
    Test_multiarray_convenience_constructor_3d_types);


// =============================================================================
// Tests on copy constructors.
TEST(TestMultiArray, test_copy_constructor)
{
  MultiArray<int, 2> A(4, 9);
  {
    MultiArray<int, 2>::iterator a = A.begin();
    int value_a = 0;
    for ( ; a != A.end(); ++a, ++value_a)
      *a = value_a;
  }

  MultiArray<int, 2> B(A);

  EXPECT_NE(A.data(), B.data());
  EXPECT_NE(A.begin(), B.begin());
  EXPECT_EQ(A.size(), B.size());
  EXPECT_MATRIX_EQ(A.sizes(), B.sizes());
  EXPECT_MATRIX_EQ(A.strides(), B.strides());

  MultiArray<int, 2>::const_iterator a = A.begin();
  MultiArray<int, 2>::const_iterator b = B.begin();
  for ( ; a != A.end(); ++a, ++b)
    ASSERT_EQ(*a, *b);
}

TEST(TestMultiArray, test_copy_constructor_from_different_multiarray_type)
{
  MultiArray<int, 2> A(4, 9);
  {
    MultiArray<int, 2>::iterator a = A.begin();
    int value_a = 0;
    for ( ; a != A.end(); ++a, ++value_a)
      *a = value_a;
  }

  MultiArray<float, 2> B(A);

  EXPECT_EQ(A.size(), B.size());
  EXPECT_MATRIX_EQ(A.sizes(), B.sizes());
  EXPECT_MATRIX_EQ(A.strides(), B.strides());

  {
    MultiArray<int, 2>::const_iterator a = A.begin();
    MultiArray<float, 2>::const_iterator b = B.begin();
    for ( ; a != A.end(); ++a, ++b)
      ASSERT_EQ(static_cast<float>(*a), *b);
  }
}

TEST(TestMultiArray, test_assignment_operator)
{
  MultiArray<int, 2> A(4, 9);
  {
    MultiArray<int, 2>::iterator a = A.begin();
    int value_a = 0;
    for ( ; a != A.end(); ++a, ++value_a)
      *a = value_a;
  }

  MultiArray<int, 2> B;
  B = A;

  EXPECT_NE(A.data(), B.data());
  EXPECT_NE(A.begin(), B.begin());
  EXPECT_EQ(A.size(), B.size());
  EXPECT_MATRIX_EQ(A.sizes(), B.sizes());
  EXPECT_MATRIX_EQ(A.strides(), B.strides());

  MultiArray<int, 2>::const_iterator a = A.begin();
  MultiArray<int, 2>::const_iterator b = B.begin();
  for ( ; a != A.end(); ++a, ++b)
    ASSERT_EQ(*a, *b);
}

TEST(TestMultiArray, test_assignment_operator_from_different_multiarray_type)
{
  MultiArray<int, 2> A(4, 9);
  {
    MultiArray<int, 2>::iterator a = A.begin();
    int value_a = 0;
    for ( ; a != A.end(); ++a, ++value_a)
      *a = value_a;
  }

  MultiArray<float, 2> B;
  B = A;

  EXPECT_EQ(A.size(), B.size());
  EXPECT_MATRIX_EQ(A.sizes(), B.sizes());
  EXPECT_MATRIX_EQ(A.strides(), B.strides());

  {
    MultiArray<int, 2>::const_iterator a = A.begin();
    MultiArray<float, 2>::const_iterator b = B.begin();
    for ( ; a != A.end(); ++a, ++b)
      ASSERT_EQ(static_cast<float>(*a), *b);
  }
}


// =============================================================================
// Test on swap operation.
TEST(TestMultiArray, test_swap)
{
  MultiArray<int, 2> A(2, 3);
  MultiArray<int, 2> B(5, 4);

  {
    MultiArray<int, 2>::iterator a = A.begin();
    int value_a = 0;
    for ( ; a != A.end(); ++a, ++value_a)
      *a = value_a;
  }

  {
    MultiArray<int, 2>::iterator b = B.begin();
    int value_b = 1;
    for ( ; b != B.end(); ++b, ++value_b)
      *b = value_b;
  }

  A.swap(B);

  {
    MultiArray<int, 2>::iterator b = B.begin();
    int value_b = 0;
    for ( ; b != B.end(); ++b, ++value_b)
      ASSERT_EQ(*b, value_b);
  }

  {
    MultiArray<int, 2>::iterator a = A.begin();
    int value_a = 1;
    for ( ; a != A.end(); ++a, ++value_a)
      ASSERT_EQ(*a, value_a);
  }
}


// =============================================================================
// Tests on overloaded operators.
template <typename MultiArrayType>
class TestOperatorOverloadingND : public testing::Test {};
TYPED_TEST_CASE_P(TestOperatorOverloadingND);
TYPED_TEST_P(TestOperatorOverloadingND, test_operator_overloading)
{
  // Read-write overloaded operators.
  typedef TypeParam VectorField;
  Vector4i sizes(3, 5, 7, 9);
  VectorField vector_field(sizes);
  for (int i = 0; i < vector_field.size(0); ++i) {
    for (int j = 0; j < vector_field.size(1); ++j) {
      for (int k = 0; k < vector_field.size(2); ++k) {
        for (int l = 0; l < vector_field.size(3); ++l) {
          Vector4i pos(i,j,k,l);
          vector_field(pos) = pos;
        }
      }
    }
  }

  // Read-only overloaded operators.
  const VectorField& const_vector_field = vector_field;
  for (int i = 0; i < const_vector_field.size(0); ++i) {
    for (int j = 0; j < const_vector_field.size(1); ++j) {
      for (int k = 0; k < const_vector_field.size(2); ++k) {
        for (int l = 0; l < const_vector_field.size(3); ++l) {
          Vector4i pos(i,j,k,l);
          ASSERT_MATRIX_EQ(const_vector_field(pos), pos);
        }
      }
    }
  }
}

REGISTER_TYPED_TEST_CASE_P(TestOperatorOverloadingND,
                           test_operator_overloading);

typedef testing::Types<
  MultiArray<Vector4i, 4, ColMajor>,
  MultiArray<Vector4i, 4, RowMajor>
> Array4DTypes;
INSTANTIATE_TYPED_TEST_CASE_P(TestMultiArray,
                              TestOperatorOverloadingND,
                              Array4DTypes);


// =============================================================================
// Tests on overloaded operators (2D case).
template <typename MultiArrayType>
class TestOperatorOverloading2D : public testing::Test {};
TYPED_TEST_CASE_P(TestOperatorOverloading2D);
TYPED_TEST_P(TestOperatorOverloading2D, test_operator_overloading)
{
  // Read-write overloaded operators.
  typedef TypeParam VectorField;
  VectorField vector_field(10, 20);
  for (int i = 0; i < vector_field.rows(); ++i)
    for (int j = 0; j < vector_field.cols(); ++j)
        vector_field(i,j) = Vector2i(i,j);

  // Read-only overloaded operators.
  const VectorField& const_vector_field = vector_field;
  for (int i = 0; i < const_vector_field.rows(); ++i)
    for (int j = 0; j < const_vector_field.cols(); ++j)
      ASSERT_MATRIX_EQ(const_vector_field(i,j), Vector2i(i,j));
}

REGISTER_TYPED_TEST_CASE_P(TestOperatorOverloading2D,
                           test_operator_overloading);

typedef testing::Types<
  MultiArray<Vector2i, 2, ColMajor>,
  MultiArray<Vector2i, 2, RowMajor>
> Array2DTypes;
INSTANTIATE_TYPED_TEST_CASE_P(TestMultiArray,
                              TestOperatorOverloading2D,
                              Array2DTypes);


// =============================================================================
// Tests on overloaded operators (3D case).
template <typename MultiArrayType>
class TestOperatorOverloading3D : public testing::Test {};
TYPED_TEST_CASE_P(TestOperatorOverloading3D);
TYPED_TEST_P(TestOperatorOverloading3D, test_operator_overloading)
{
  typedef TypeParam VectorField;
  VectorField vector_field(5, 10, 20);
  for (int i = 0; i < vector_field.rows(); ++i)
    for (int j = 0; j < vector_field.cols(); ++j)
      for (int k = 0; k < vector_field.depth(); ++k)
        vector_field(i,j,k) = Vector3i(i,j,k);

  const VectorField& const_vector_field = vector_field;
  for (int i = 0; i < const_vector_field.rows(); ++i)
    for (int j = 0; j < const_vector_field.cols(); ++j)
      for (int k = 0; k < const_vector_field.depth(); ++k)
        ASSERT_MATRIX_EQ(const_vector_field(i,j,k), Vector3i(i,j,k));
}

REGISTER_TYPED_TEST_CASE_P(TestOperatorOverloading3D,
                           test_operator_overloading);

typedef testing::Types<
  MultiArray<Vector3i, 3, ColMajor>,
  MultiArray<Vector3i, 3, RowMajor>
> Array3DTypes;
INSTANTIATE_TYPED_TEST_CASE_P(TestMultiArray,
                              TestOperatorOverloading3D,
                              Array3DTypes);


// =============================================================================
// Tests on resize operator.
TEST(TestMultiArray, test_resize)
{
  typedef MultiArray<char, 4> Array4;
  Array4 multi_array;
  EXPECT_FALSE(multi_array.data());
  EXPECT_FALSE(multi_array.begin());
  EXPECT_FALSE(multi_array.end());
  EXPECT_MATRIX_EQ(multi_array.sizes(), Vector4i::Zero().eval());
  EXPECT_MATRIX_EQ(multi_array.strides(), Vector4i::Zero().eval());

  Vector4i sizes(2, 3, 4, 5);
  multi_array.resize(sizes);
  EXPECT_TRUE(multi_array.data() != 0);
  EXPECT_TRUE(multi_array.begin() != 0);
  EXPECT_TRUE(multi_array.end() != 0);
  EXPECT_MATRIX_EQ(multi_array.sizes(), sizes);
  EXPECT_MATRIX_EQ(
    multi_array.strides(),
    StrideComputer<Array4::StorageOrder>::eval(sizes));
}

TEST(TestMultiArray, test_resize_2d)
{
  typedef MultiArray<char, 2> Array2;
  Array2 multi_array;
  EXPECT_FALSE(multi_array.data());
  EXPECT_FALSE(multi_array.begin());
  EXPECT_FALSE(multi_array.end());
  EXPECT_MATRIX_EQ(multi_array.sizes(), Vector2i::Zero().eval());
  EXPECT_MATRIX_EQ(multi_array.strides(), Vector2i::Zero().eval());

  multi_array.resize(10, 20);
  EXPECT_TRUE(multi_array.data() != 0);
  EXPECT_TRUE(multi_array.begin() != 0);
  EXPECT_TRUE(multi_array.end() != 0);
  EXPECT_MATRIX_EQ(multi_array.sizes(), Vector2i(10, 20));
  EXPECT_MATRIX_EQ(
    multi_array.strides(),
    StrideComputer<Array2::StorageOrder>::eval(Vector2i(10, 20)));
}

TEST(TestMultiArray, test_resize_3d)
{
  typedef MultiArray<char, 3> Array3;
  Array3 multi_array;
  EXPECT_FALSE(multi_array.data());
  EXPECT_FALSE(multi_array.begin());
  EXPECT_FALSE(multi_array.end());
  EXPECT_MATRIX_EQ(multi_array.sizes(), Vector3i::Zero().eval());
  EXPECT_MATRIX_EQ(multi_array.strides(), Vector3i::Zero().eval());

  multi_array.resize(2, 3, 5);
  EXPECT_TRUE(multi_array.data() != 0);
  EXPECT_TRUE(multi_array.begin() != 0);
  EXPECT_TRUE(multi_array.end() != 0);
  EXPECT_MATRIX_EQ(multi_array.sizes(), Vector3i(2, 3, 5));
  EXPECT_MATRIX_EQ(
    multi_array.strides(),
    StrideComputer<Array3::StorageOrder>::eval(Vector3i(2, 3, 5)));
}


// =============================================================================
// Tests on array and matrix views.
TEST(TestMultiArray, test_array_view_types)
{
  typedef MultiArray<int, 2> Array2;
  Array2 A(2,2);
  A.array() << 1, 2, 3, 4;

  Array4i actual_A;
  Array4i expected_A;
  actual_A = A.array();
  expected_A << 1, 2, 3, 4;

  for (int i = 0; i < actual_A.size(); ++i)
    ASSERT_EQ(actual_A(i), expected_A(i));
}

TEST(TestMultiArray, test_matrix_view_types)
{
  typedef MultiArray<float, 2> Matrix2;
  Matrix2 A(2,2);
  A.matrix() << 1, 2,
                3, 4;
  Matrix2f actual_A = A.matrix();

  Matrix2f expected_A;
  expected_A << 1, 2,
                3, 4;

  EXPECT_MATRIX_EQ(actual_A, expected_A);

  EXPECT_EQ(A(0,0), 1);
  EXPECT_EQ(A(0,1), 2);
  EXPECT_EQ(A(1,0), 3);
  EXPECT_EQ(A(1,1), 4);
}


// =============================================================================
// Test runner.
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
