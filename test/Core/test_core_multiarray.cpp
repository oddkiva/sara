// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <vector>

#include <gtest/gtest.h>

#include <DO/Sara/Core/MultiArray.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


// =============================================================================
// Tests on generic constructors.
template <typename MultiArrayType>
class Test_multiarray_constructors : public testing::Test {};
TYPED_TEST_CASE_P(Test_multiarray_constructors);

TYPED_TEST_P(Test_multiarray_constructors,
             test_default_constructor_and_base_getters)
{
  using MultiArray = TypeParam;
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
  using MultiArray = TypeParam;
  using Vector = typename MultiArray::vector_type;
  using StrideComputer = StrideComputer<MultiArray::StorageOrder>;

  // Create sizes, one for each dimension.
  auto sizes = Vector{};
  for (auto i = 0; i < sizes.size(); ++i)
    sizes(i) = (i+1)*10;

  // Compute the strides.
  Vector strides = StrideComputer::eval(sizes);

  // Compute the raw size of the multi-array.
  auto raw_size = static_cast<size_t>( accumulate(
    sizes.data(), sizes.data() + sizes.size(), 1,
    multiplies<int>()) );
  // Create the array with the right sizes.
  auto multi_array = MultiArray{ sizes };

  EXPECT_EQ(multi_array.sizes(), sizes);
  EXPECT_EQ(multi_array.strides(), strides);
  EXPECT_EQ(multi_array.size(), raw_size);

  // Check size values in each dimension.
  for (auto i = 0; i < sizes.size(); ++i)
    ASSERT_EQ(multi_array.size(i), sizes(i));
  // Check stride values in each dimension.
  for (auto i = 0; i < sizes.size(); ++i)
    ASSERT_EQ(multi_array.stride(i), strides(i));
}

REGISTER_TYPED_TEST_CASE_P(Test_multiarray_constructors,
                           test_default_constructor_and_base_getters,
                           test_constructor_with_sizes_and_base_getters);

using Test_multiarray_constructors_types = testing::Types<
  MultiArray<unsigned char, 2, ColMajor>,
  MultiArray<unsigned char, 2, RowMajor>,
  MultiArray<unsigned char, 3, ColMajor>,
  MultiArray<unsigned char, 3, RowMajor>,
  MultiArray<unsigned char, 4, ColMajor>,
  MultiArray<unsigned char, 4, RowMajor>,
  MultiArray<unsigned char, 5, ColMajor>,
  MultiArray<unsigned char, 5, RowMajor>
>;
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
  using MultiArray = TypeParam;
  using StrideComputer = StrideComputer<MultiArray::StorageOrder>;

  MultiArray multi_array{ 10, 20 };

  EXPECT_EQ(multi_array.sizes(), Vector2i(10, 20));
  EXPECT_EQ(multi_array.rows(), 10);
  EXPECT_EQ(multi_array.cols(), 20);
  EXPECT_EQ(multi_array.strides(), StrideComputer::eval(Vector2i(10, 20)));
  EXPECT_EQ(multi_array.size(), 10u*20u);
}
REGISTER_TYPED_TEST_CASE_P(
  Test_multiarray_convenience_constructor_2d_and_getters, run);

using Test_multiarray_convenience_constructor_2d_types = testing::Types<
  MultiArray<int, 2, ColMajor>,
  MultiArray<int, 2, RowMajor>,
  MultiArray<float, 2, ColMajor>,
  MultiArray<float, 2, RowMajor>
>;
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
  using MultiArray = TypeParam;
  using StrideComputer = StrideComputer<MultiArray::StorageOrder>;

  MultiArray multi_array{ 10, 20, 50 };

  EXPECT_EQ(multi_array.sizes(), Vector3i(10, 20, 50));
  EXPECT_EQ(multi_array.rows(), 10);
  EXPECT_EQ(multi_array.cols(), 20);
  EXPECT_EQ(multi_array.depth(), 50);
  EXPECT_EQ(multi_array.strides(), StrideComputer::eval(Vector3i(10, 20, 50)));
  EXPECT_EQ(multi_array.size(), 10u*20u*50u);
}
REGISTER_TYPED_TEST_CASE_P(
  Test_multiarray_convenience_constructor_3d_and_getters, run);

using Test_multiarray_convenience_constructor_3d_types = testing::Types<
  MultiArray<int, 3, ColMajor>,
  MultiArray<int, 3, RowMajor>,
  MultiArray<float, 3, ColMajor>,
  MultiArray<float, 3, RowMajor>
>;
INSTANTIATE_TYPED_TEST_CASE_P(
  TestMultiArray,
  Test_multiarray_convenience_constructor_3d_and_getters,
  Test_multiarray_convenience_constructor_3d_types);


// =============================================================================
// Tests on copy constructors.
TEST(TestMultiArray, test_copy_constructor)
{
  auto A = MultiArray<int, 2>{ 4, 9 };
  {
    auto value_a = 0;
    for (auto a = A.begin() ; a != A.end(); ++a, ++value_a)
      *a = value_a;
  }

  auto B = A;

  EXPECT_NE(A.data(), B.data());
  EXPECT_NE(A.begin(), B.begin());
  EXPECT_EQ(A.size(), B.size());
  EXPECT_MATRIX_EQ(A.sizes(), B.sizes());
  EXPECT_MATRIX_EQ(A.strides(), B.strides());

  auto a = A.begin();
  auto b = B.begin();
  for ( ; a != A.end(); ++a, ++b)
    ASSERT_EQ(*a, *b);
}

TEST(TestMultiArray, test_assignment_operator)
{
  auto A = MultiArray<int, 2>{ 4, 9 };
  {
    auto value_a = 0;
    for (auto a = A.begin(); a != A.end(); ++a, ++value_a)
      *a = value_a;
  }

  auto B = MultiArray<int, 2>{};
  B = A;

  EXPECT_NE(A.data(), B.data());
  EXPECT_NE(A.begin(), B.begin());
  EXPECT_EQ(A.size(), B.size());
  EXPECT_MATRIX_EQ(A.sizes(), B.sizes());
  EXPECT_MATRIX_EQ(A.strides(), B.strides());

  auto a = A.begin();
  auto b = B.begin();
  for ( ; a != A.end(); ++a, ++b)
    ASSERT_EQ(*a, *b);
}


// =============================================================================
// Test on swap operation.
TEST(TestMultiArray, test_swap)
{
  auto A = MultiArray<int, 2>{ 2, 3 };
  auto B = MultiArray<int, 2>{ 5, 4 };

  {
    auto value_a = 0;
    for (auto a = A.begin(); a != A.end(); ++a, ++value_a)
      *a = value_a;
  }

  {
    auto value_b = 1;
    for (auto b = B.begin(); b != B.end(); ++b, ++value_b)
      *b = value_b;
  }

  A.swap(B);

  {
    auto value_b = 0;
    for (auto b = B.begin(); b != B.end(); ++b, ++value_b)
      ASSERT_EQ(*b, value_b);
  }

  {
    auto value_a = 1;
    for (auto a = A.begin(); a != A.end(); ++a, ++value_a)
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
  using VectorField = TypeParam;
  auto sizes = Vector4i{ 3, 5, 7, 9 };
  auto vector_field = VectorField{ sizes };
  for (auto i = 0; i < vector_field.size(0); ++i)
  {
    for (auto j = 0; j < vector_field.size(1); ++j)
    {
      for (auto k = 0; k < vector_field.size(2); ++k)
      {
        for (auto l = 0; l < vector_field.size(3); ++l)
        {
          auto pos = Vector4i{ i, j, k, l };
          vector_field(pos) = pos;
        }
      }
    }
  }

  // Read-only overloaded operators.
  const auto& const_vector_field = vector_field;
  for (auto i = 0; i < vector_field.size(0); ++i)
  {
    for (auto j = 0; j < vector_field.size(1); ++j)
    {
      for (auto k = 0; k < vector_field.size(2); ++k)
      {
        for (auto l = 0; l < vector_field.size(3); ++l)
        {
          auto pos = Vector4i{ i, j, k, l };
          ASSERT_MATRIX_EQ(const_vector_field(pos), pos);
        }
      }
    }
  }
}

REGISTER_TYPED_TEST_CASE_P(TestOperatorOverloadingND,
                           test_operator_overloading);

using Array4DTypes = testing::Types<
  MultiArray<Vector4i, 4, ColMajor>,
  MultiArray<Vector4i, 4, RowMajor>
>;
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
  using VectorField = TypeParam;
  auto vector_field = VectorField{ 10, 20 };
  for (auto i = 0; i < vector_field.rows(); ++i)
    for (auto j = 0; j < vector_field.cols(); ++j)
        vector_field(i,j) = Vector2i(i,j);

  // Read-only overloaded operators.
  const auto& const_vector_field = vector_field;
  for (auto i = 0; i < const_vector_field.rows(); ++i)
    for (auto j = 0; j < const_vector_field.cols(); ++j)
      ASSERT_MATRIX_EQ(const_vector_field(i,j), Vector2i(i,j));
}

REGISTER_TYPED_TEST_CASE_P(TestOperatorOverloading2D,
                           test_operator_overloading);

using Array2DTypes = testing::Types<
  MultiArray<Vector2i, 2, ColMajor>,
  MultiArray<Vector2i, 2, RowMajor>
>;
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
  using VectorField = TypeParam;
  auto vector_field = VectorField{ 5, 10, 20 };
  for (auto i = 0; i < vector_field.rows(); ++i)
    for (auto j = 0; j < vector_field.cols(); ++j)
      for (auto k = 0; k < vector_field.depth(); ++k)
        vector_field(i,j,k) = Vector3i(i,j,k);

  const VectorField& const_vector_field = vector_field;
  for (auto i = 0; i < const_vector_field.rows(); ++i)
    for (auto j = 0; j < const_vector_field.cols(); ++j)
      for (auto k = 0; k < const_vector_field.depth(); ++k)
        ASSERT_MATRIX_EQ(const_vector_field(i,j,k), Vector3i(i,j,k));
}

REGISTER_TYPED_TEST_CASE_P(TestOperatorOverloading3D,
                           test_operator_overloading);

using Array3DTypes = testing::Types<
  MultiArray<Vector3i, 3, ColMajor>,
  MultiArray<Vector3i, 3, RowMajor>
>;
INSTANTIATE_TYPED_TEST_CASE_P(TestMultiArray,
                              TestOperatorOverloading3D,
                              Array3DTypes);


// =============================================================================
// Tests on resize operator.
TEST(TestMultiArray, test_resize)
{
  using Array4 = MultiArray<char, 4>;
  auto multi_array = Array4{};
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
  using Array2 = MultiArray<char, 2>;
  auto multi_array = Array2{};
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
  using Array3 = MultiArray<char, 3>;
  auto multi_array = Array3{};
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
  using Array2 = MultiArray<int, 2>;
  auto A = Array2{ 2, 2 };
  A.array() << 1, 2, 3, 4;

  auto actual_A = Array4i{};
  auto expected_A = Array4i{};
  actual_A = A.array();
  expected_A << 1, 2, 3, 4;

  for (auto i = 0; i < actual_A.size(); ++i)
    ASSERT_EQ(actual_A(i), expected_A(i));
}

TEST(TestMultiArray, test_matrix_view_types)
{
  using Matrix2 = MultiArray<float, 2>;
  auto A = Matrix2{ 2, 2 };
  A.matrix() << 1, 2,
                3, 4;
  auto actual_A = A.matrix();

  auto expected_A = Matrix2f{};
  expected_A << 1, 2,
                3, 4;

  EXPECT_MATRIX_EQ(actual_A, expected_A);

  EXPECT_EQ(A(0,0), 1);
  EXPECT_EQ(A(0,1), 2);
  EXPECT_EQ(A(1,0), 3);
  EXPECT_EQ(A(1,1), 4);
}


TEST(TestMultiArray, test_slices)
{
  using Vector5i = Matrix<int, 5, 1>;
  using RgbHistogramBlocks = MultiArray<float, 5, RowMajor>;
  using vector_type = RgbHistogramBlocks::vector_type;

  const auto block_sizes = Vector2i{ 4, 4 };
  const Vector3i histogram_sizes{ 6, 6, 6 };
  auto sizes = vector_type{};
  sizes << block_sizes, histogram_sizes;

  auto rgb_histogram_blocks = RgbHistogramBlocks{ sizes };
  for (auto i = 0; i < rgb_histogram_blocks.size(0); ++i)
  {
    for (auto j = 0; j < rgb_histogram_blocks.size(1); ++j)
    {
      auto pos = Vector5i{};
      pos << i, j, 0, 0, 0;

      // Read-only view of multi-array.
      const auto const_rgb_histogram = rgb_histogram_blocks[i][j];
      ASSERT_EQ(const_rgb_histogram.sizes(), const_rgb_histogram.sizes());
      ASSERT_EQ(const_rgb_histogram.strides(), Vector3i(36, 6, 1));
      ASSERT_EQ(rgb_histogram_blocks.data() + i * 4 * 6 * 6 * 6 + j * 6 * 6 * 6,
                const_rgb_histogram.data());

      // Read-write view of multi-array.
      auto rgb_histogram = rgb_histogram_blocks[i][j];
      for (int r = 0; r < rgb_histogram.size(0); ++r)
        for (int g = 0; g < rgb_histogram.size(1); ++g)
          for (int b = 0; b < rgb_histogram.size(2); ++b)
            rgb_histogram(r, g, b) = static_cast<float>(i*rgb_histogram_blocks.size(1) + j);
    }
  }

  for (auto i = 0; i < rgb_histogram_blocks.size(0); ++i)
    for (auto j = 0; j < rgb_histogram_blocks.size(1); ++j)
      for (auto r = 0; r < rgb_histogram_blocks.size(2); ++r)
        for (auto g = 0; g < rgb_histogram_blocks.size(3); ++g)
          for (auto b = 0; b < rgb_histogram_blocks.size(4); ++b)
          {
            auto pos = Vector5i{};
            pos << i, j, r, g, b;
            ASSERT_EQ(rgb_histogram_blocks(pos),
              i*rgb_histogram_blocks.size(1) + j);
          }
}

//
TEST(TestMultiArray, test_clear)
{
  auto array = MultiArray<int, 2>{ 300, 300 };
  EXPECT_MATRIX_EQ(Vector2i(300, 300), array.sizes());

  array.clear();
  EXPECT_MATRIX_EQ(Vector2i::Zero(), array.sizes());
  EXPECT_EQ(nullptr, array.data());
  EXPECT_EQ(nullptr, array.begin());
  EXPECT_EQ(nullptr, array.end());
}

// =============================================================================
// Test runner.
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
