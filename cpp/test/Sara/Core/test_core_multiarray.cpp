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

#define BOOST_TEST_MODULE "Core/MultiArray/MultiArray Class"

#include <vector>

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/MultiArray.hpp>


using namespace std;
using namespace DO::Sara;


// =============================================================================
// Tests on generic constructors.
using Test_multiarray_constructors_types =
    boost::mpl::list<MultiArray<unsigned char, 2, ColMajor>,
                     MultiArray<unsigned char, 2, RowMajor>,
                     MultiArray<unsigned char, 3, ColMajor>,
                     MultiArray<unsigned char, 3, RowMajor>,
                     MultiArray<unsigned char, 4, ColMajor>,
                     MultiArray<unsigned char, 4, RowMajor>,
                     MultiArray<unsigned char, 5, ColMajor>,
                     MultiArray<unsigned char, 5, RowMajor>>;

BOOST_AUTO_TEST_SUITE(TestMultiArrayConstructors)

BOOST_AUTO_TEST_CASE_TEMPLATE(test_default_constructor_and_base_getters,
                              MultiArray, Test_multiarray_constructors_types)
{
  auto multi_array = MultiArray{};
  BOOST_CHECK(multi_array.data() == nullptr);
  BOOST_CHECK(multi_array.begin() == nullptr);
  BOOST_CHECK(multi_array.end() == nullptr);
  BOOST_CHECK_EQUAL(multi_array.sizes(),
                    MultiArray::vector_type::Zero().eval());
  BOOST_CHECK_EQUAL(multi_array.strides(),
                    MultiArray::vector_type::Zero().eval());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_constructor_with_sizes_and_base_getters,
                              MultiArray, Test_multiarray_constructors_types)
{
  using Vector = typename MultiArray::vector_type;
  using StrideComputer = StrideComputer<MultiArray::StorageOrder>;

  // Create sizes, one for each dimension.
  auto sizes = Vector{};
  for (auto i = 0; i < sizes.size(); ++i)
    sizes(i) = (i + 1) * 10;

  // Compute the strides.
  Vector strides = StrideComputer::eval(sizes);

  // Compute the raw size of the multi-array.
  auto raw_size = static_cast<size_t>(accumulate(
      sizes.data(), sizes.data() + sizes.size(), 1, multiplies<int>()));
  // Create the array with the right sizes.
  auto multi_array = MultiArray{sizes};

  BOOST_CHECK_EQUAL(multi_array.sizes(), sizes);
  BOOST_CHECK_EQUAL(multi_array.strides(), strides);
  BOOST_CHECK_EQUAL(multi_array.size(), raw_size);

  // Check size values in each dimension.
  for (auto i = 0; i < sizes.size(); ++i)
    BOOST_REQUIRE_EQUAL(multi_array.size(i), sizes(i));
  // Check stride values in each dimension.
  for (auto i = 0; i < sizes.size(); ++i)
    BOOST_REQUIRE_EQUAL(multi_array.stride(i), strides(i));
}


// =============================================================================
// Tests on convenience constructor (2D case)
using Test_multiarray_convenience_constructor_2d_types =
    boost::mpl::list<MultiArray<int, 2, ColMajor>, MultiArray<int, 2, RowMajor>,
                     MultiArray<float, 2, ColMajor>,
                     MultiArray<float, 2, RowMajor>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(
    test_multiarray_convenience_constructor_2d_and_getters, MultiArray,
    Test_multiarray_convenience_constructor_2d_types)
{
  using StrideComputer = StrideComputer<MultiArray::StorageOrder>;

  const auto multi_array = MultiArray{10, 20};

  BOOST_CHECK_EQUAL(multi_array.sizes(), Vector2i(10, 20));
  BOOST_CHECK_EQUAL(multi_array.rows(), 10);
  BOOST_CHECK_EQUAL(multi_array.cols(), 20);
  BOOST_CHECK_EQUAL(multi_array.strides(),
                    StrideComputer::eval(Vector2i(10, 20)));
  BOOST_CHECK_EQUAL(multi_array.size(), 10u * 20u);
}


// =============================================================================
// Tests on convenience constructor (3D case)
using Test_multiarray_convenience_constructor_3d_types =
    boost::mpl::list<MultiArray<int, 3, ColMajor>, MultiArray<int, 3, RowMajor>,
                     MultiArray<float, 3, ColMajor>,
                     MultiArray<float, 3, RowMajor>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(
    test_multiarray_convenience_constructor_3d_and_getters, MultiArray,
    Test_multiarray_convenience_constructor_3d_types)
{
  using StrideComputer = StrideComputer<MultiArray::StorageOrder>;

  const auto multi_array = MultiArray{10, 20, 50};

  BOOST_CHECK_EQUAL(multi_array.sizes(), Vector3i(10, 20, 50));
  BOOST_CHECK_EQUAL(multi_array.rows(), 10);
  BOOST_CHECK_EQUAL(multi_array.cols(), 20);
  BOOST_CHECK_EQUAL(multi_array.depth(), 50);
  BOOST_CHECK_EQUAL(multi_array.strides(),
                    StrideComputer::eval(Vector3i(10, 20, 50)));
  BOOST_CHECK_EQUAL(multi_array.size(), 10u * 20u * 50u);
}


// =============================================================================
// Tests on copy constructors.
BOOST_AUTO_TEST_CASE(test_copy_constructor)
{
  auto A = MultiArray<int, 2>{4, 9};
  {
    auto value_a = 0;
    for (auto a = A.begin(); a != A.end(); ++a, ++value_a)
      *a = value_a;
  }

  auto B = A;

  BOOST_CHECK(A.data() != B.data());
  BOOST_CHECK(A.begin() != B.begin());
  BOOST_CHECK_EQUAL(A.size(), B.size());
  BOOST_CHECK_EQUAL(A.sizes(), B.sizes());
  BOOST_CHECK_EQUAL(A.strides(), B.strides());

  auto a = A.begin();
  auto b = B.begin();
  for (; a != A.end(); ++a, ++b)
    BOOST_REQUIRE_EQUAL(*a, *b);
}

BOOST_AUTO_TEST_CASE(test_assignment_operator)
{
  auto A = MultiArray<int, 2>{4, 9};
  {
    auto value_a = 0;
    for (auto a = A.begin(); a != A.end(); ++a, ++value_a)
      *a = value_a;
  }

  auto B = MultiArray<int, 2>{};
  B = A;

  BOOST_CHECK(A.data() != B.data());
  BOOST_CHECK(A.begin() != B.begin());
  BOOST_CHECK_EQUAL(A.size(), B.size());
  BOOST_CHECK_EQUAL(A.sizes(), B.sizes());
  BOOST_CHECK_EQUAL(A.strides(), B.strides());

  auto a = A.begin();
  auto b = B.begin();
  for (; a != A.end(); ++a, ++b)
    BOOST_REQUIRE_EQUAL(*a, *b);
}


// =============================================================================
// Test on swap operation.
BOOST_AUTO_TEST_CASE(test_swap)
{
  auto A = MultiArray<int, 2>{2, 3};
  auto B = MultiArray<int, 2>{5, 4};

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
      BOOST_REQUIRE_EQUAL(*b, value_b);
  }

  {
    auto value_a = 1;
    for (auto a = A.begin(); a != A.end(); ++a, ++value_a)
      BOOST_REQUIRE_EQUAL(*a, value_a);
  }
}


// =============================================================================
// Tests on overloaded operators (2D case).
using Array2DTypes = boost::mpl::list<MultiArray<Vector2i, 2, ColMajor>,
                                      MultiArray<Vector2i, 2, RowMajor>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_operator_overloading_2d, VectorField,
                              Array2DTypes)
{
  // Read-write overloaded operators.
  auto vector_field = VectorField{10, 20};
  for (auto i = 0; i < vector_field.rows(); ++i)
    for (auto j = 0; j < vector_field.cols(); ++j)
      vector_field(i, j) = Vector2i(i, j);

  // Read-only overloaded operators.
  const auto& const_vector_field = vector_field;
  for (auto i = 0; i < const_vector_field.rows(); ++i)
    for (auto j = 0; j < const_vector_field.cols(); ++j)
      BOOST_REQUIRE_EQUAL(const_vector_field(i, j), Vector2i(i, j));
}


// =============================================================================
// Tests on overloaded operators (3D case).
using Array3DTypes = boost::mpl::list<MultiArray<Vector3i, 3, ColMajor>,
                                      MultiArray<Vector3i, 3, RowMajor>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_operator_overloading_3d, VectorField,
                              Array3DTypes)
{
  auto vector_field = VectorField{5, 10, 20};
  for (auto i = 0; i < vector_field.rows(); ++i)
    for (auto j = 0; j < vector_field.cols(); ++j)
      for (auto k = 0; k < vector_field.depth(); ++k)
        vector_field(i, j, k) = Vector3i(i, j, k);

  const VectorField& const_vector_field = vector_field;
  for (auto i = 0; i < const_vector_field.rows(); ++i)
    for (auto j = 0; j < const_vector_field.cols(); ++j)
      for (auto k = 0; k < const_vector_field.depth(); ++k)
        BOOST_REQUIRE_EQUAL(const_vector_field(i, j, k), Vector3i(i, j, k));
}


// =============================================================================
// Tests on overloaded operators (4D case).
using Array4DTypes = boost::mpl::list<MultiArray<Vector4i, 4, ColMajor>,
                                      MultiArray<Vector4i, 4, RowMajor>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_operator_overloading_4d, VectorField,
                              Array4DTypes)
{
  // Read-write overloaded operators.
  auto sizes = Vector4i{3, 5, 7, 9};
  auto vector_field = VectorField{sizes};
  for (auto i = 0; i < vector_field.size(0); ++i)
  {
    for (auto j = 0; j < vector_field.size(1); ++j)
    {
      for (auto k = 0; k < vector_field.size(2); ++k)
      {
        for (auto l = 0; l < vector_field.size(3); ++l)
        {
          auto pos = Vector4i{i, j, k, l};
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
          auto pos = Vector4i{i, j, k, l};
          BOOST_REQUIRE_EQUAL(const_vector_field(pos), pos);
        }
      }
    }
  }
}

// =============================================================================
// Tests on resize operator.
BOOST_AUTO_TEST_CASE(test_resize)
{
  using Array4 = MultiArray<char, 4>;
  auto multi_array = Array4{};
  BOOST_CHECK(multi_array.data() == nullptr);
  BOOST_CHECK(multi_array.begin() == nullptr);
  BOOST_CHECK(multi_array.end() == nullptr);
  BOOST_CHECK_EQUAL(multi_array.sizes(), Vector4i::Zero().eval());
  BOOST_CHECK_EQUAL(multi_array.strides(), Vector4i::Zero().eval());

  Vector4i sizes(2, 3, 4, 5);
  multi_array.resize(sizes);
  BOOST_CHECK(multi_array.data() != nullptr);
  BOOST_CHECK(multi_array.begin() != nullptr);
  BOOST_CHECK(multi_array.end() != nullptr);
  BOOST_CHECK_EQUAL(multi_array.sizes(), sizes);
  BOOST_CHECK_EQUAL(multi_array.strides(),
                    StrideComputer<Array4::StorageOrder>::eval(sizes));
}

BOOST_AUTO_TEST_CASE(test_resize_2d)
{
  using Array2 = MultiArray<char, 2>;
  auto multi_array = Array2{};
  BOOST_CHECK(multi_array.data() == nullptr);
  BOOST_CHECK(multi_array.begin() == nullptr);
  BOOST_CHECK(multi_array.end() == nullptr);
  BOOST_CHECK_EQUAL(multi_array.sizes(), Vector2i::Zero().eval());
  BOOST_CHECK_EQUAL(multi_array.strides(), Vector2i::Zero().eval());

  multi_array.resize(10, 20);
  BOOST_CHECK(multi_array.data() != nullptr);
  BOOST_CHECK(multi_array.begin() != nullptr);
  BOOST_CHECK(multi_array.end() != nullptr);
  BOOST_CHECK_EQUAL(multi_array.sizes(), Vector2i(10, 20));
  BOOST_CHECK_EQUAL(
      multi_array.strides(),
      StrideComputer<Array2::StorageOrder>::eval(Vector2i(10, 20)));
}

BOOST_AUTO_TEST_CASE(test_resize_3d)
{
  using Array3 = MultiArray<char, 3>;
  auto multi_array = Array3{};
  BOOST_CHECK(multi_array.data() == nullptr);
  BOOST_CHECK(multi_array.begin() == nullptr);
  BOOST_CHECK(multi_array.end() == nullptr);
  BOOST_CHECK_EQUAL(multi_array.sizes(), Vector3i::Zero().eval());
  BOOST_CHECK_EQUAL(multi_array.strides(), Vector3i::Zero().eval());

  multi_array.resize(2, 3, 5);
  BOOST_CHECK(multi_array.data() != nullptr);
  BOOST_CHECK(multi_array.begin() != nullptr);
  BOOST_CHECK(multi_array.end() != nullptr);
  BOOST_CHECK_EQUAL(multi_array.sizes(), Vector3i(2, 3, 5));
  BOOST_CHECK_EQUAL(
      multi_array.strides(),
      StrideComputer<Array3::StorageOrder>::eval(Vector3i(2, 3, 5)));
}


// =============================================================================
// Tests on array and matrix views.
BOOST_AUTO_TEST_CASE(test_array_view_types)
{
  using Array2 = MultiArray<int, 2>;
  auto A = Array2{2, 2};
  A.flat_array() << 1, 2, 3, 4;

  auto actual_A = Array4i{};
  auto expected_A = Array4i{};
  actual_A = A.flat_array();
  expected_A << 1, 2, 3, 4;

  for (auto i = 0; i < actual_A.size(); ++i)
    BOOST_REQUIRE_EQUAL(actual_A(i), expected_A(i));
}

BOOST_AUTO_TEST_CASE(test_matrix_view_types)
{
  using Matrix2 = MultiArray<float, 2>;
  auto A = Matrix2{2, 2};
  A.matrix() << 1, 2, 3, 4;
  auto actual_A = A.matrix();

  auto expected_A = Matrix2f{};
  expected_A << 1, 2, 3, 4;

  BOOST_CHECK_EQUAL(actual_A, expected_A);

  BOOST_CHECK_EQUAL(A(0, 0), 1);
  BOOST_CHECK_EQUAL(A(0, 1), 2);
  BOOST_CHECK_EQUAL(A(1, 0), 3);
  BOOST_CHECK_EQUAL(A(1, 1), 4);
}

BOOST_AUTO_TEST_CASE(test_slices)
{
  using Vector5i = Matrix<int, 5, 1>;
  using RgbHistogramBlocks = MultiArray<float, 5, RowMajor>;
  using vector_type = RgbHistogramBlocks::vector_type;

  const auto block_sizes = Vector2i{4, 4};
  const Vector3i histogram_sizes{6, 6, 6};
  auto sizes = vector_type{};
  sizes << block_sizes, histogram_sizes;

  auto rgb_histogram_blocks = RgbHistogramBlocks{sizes};
  for (auto i = 0; i < rgb_histogram_blocks.size(0); ++i)
  {
    for (auto j = 0; j < rgb_histogram_blocks.size(1); ++j)
    {
      auto pos = Vector5i{};
      pos << i, j, 0, 0, 0;

      // Read-only view of multi-array.
      const auto const_rgb_histogram = rgb_histogram_blocks[i][j];
      BOOST_REQUIRE_EQUAL(const_rgb_histogram.sizes(),
                          const_rgb_histogram.sizes());
      BOOST_REQUIRE_EQUAL(const_rgb_histogram.strides(), Vector3i(36, 6, 1));
      BOOST_REQUIRE_EQUAL(rgb_histogram_blocks.data() + i * 4 * 6 * 6 * 6 +
                              j * 6 * 6 * 6,
                          const_rgb_histogram.data());

      // Read-write view of multi-array.
      auto rgb_histogram = rgb_histogram_blocks[i][j];
      for (int r = 0; r < rgb_histogram.size(0); ++r)
        for (int g = 0; g < rgb_histogram.size(1); ++g)
          for (int b = 0; b < rgb_histogram.size(2); ++b)
            rgb_histogram(r, g, b) =
                static_cast<float>(i * rgb_histogram_blocks.size(1) + j);
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
            BOOST_REQUIRE_EQUAL(rgb_histogram_blocks(pos),
                                i * rgb_histogram_blocks.size(1) + j);
          }
}

BOOST_AUTO_TEST_CASE(test_clear)
{
  auto array = MultiArray<int, 2>{300, 300};
  BOOST_CHECK_EQUAL(Vector2i(300, 300), array.sizes());

  array.clear();
  BOOST_CHECK_EQUAL(Vector2i::Zero(), array.sizes());
  BOOST_CHECK(nullptr == array.data());
  BOOST_CHECK(nullptr == array.begin());
  BOOST_CHECK(nullptr == array.end());
}


BOOST_AUTO_TEST_CASE(test_equality_comparison)
{
  int data[] = {0, 1, 2, 3};
  auto sizes = Vector2i{2, 2};

  {
    auto matrix_view = MultiArrayView<int, 2, RowMajor>{data, sizes};

    auto m1 = MultiArray<int, 2, RowMajor>{sizes};
    m1.matrix() << 0, 1, 2, 3;

    // Be careful of this one!
    auto m2 = MultiArray<int, 2, RowMajor>{sizes};
    m2.matrix() << 0, 2, 1, 3;

    // Different sizes.
    auto m3 = MultiArray<int, 2, RowMajor>{3, 2};
    m3.matrix() << 0, 0, 0, 1, 2, 3;

    BOOST_CHECK(matrix_view == m1);
    BOOST_CHECK(matrix_view != m2);
    BOOST_CHECK(matrix_view != m3);
  }

  {
    auto matrix_view = MultiArrayView<int, 2, ColMajor>{data, sizes};

    // What we should read.
    auto m1 = MultiArray<int, 2, ColMajor>{sizes};
    m1.matrix() << 0, 2, 1, 3;

    // Be careful of this one!
    auto m2 = MultiArray<int, 2, ColMajor>{sizes};
    m2.matrix() << 0, 1, 2, 3;

    // Different sizes.
    auto m3 = MultiArray<int, 2, ColMajor>{3, 2};
    m3.matrix() << 0, 0, 0, 1, 2, 3;

    BOOST_CHECK(matrix_view == m1);
    BOOST_CHECK(matrix_view != m2);
    BOOST_CHECK(matrix_view != m3);
  }
}

BOOST_AUTO_TEST_CASE(test_cwise_transform)
{
  auto m = MultiArray<int, 2>{2, 2};
  m.matrix() << 0, 2, 1, 3;

  auto result =
      m.cwise_transform([](int x) -> Vector3i { return Vector3i::Ones() * x; });

  BOOST_CHECK_EQUAL(Vector3i::Zero(), result(0, 0));
  BOOST_CHECK_EQUAL(Vector3i::Ones() * 1, result(1, 0));
  BOOST_CHECK_EQUAL(Vector3i::Ones() * 2, result(0, 1));
  BOOST_CHECK_EQUAL(Vector3i::Ones() * 3, result(1, 1));
}

BOOST_AUTO_TEST_CASE(test_cwise_transform_inplace)
{
  auto m = MultiArray<int, 2>{2, 2};
  m.matrix() << 0, 4, 2, 6;

  m.cwise_transform_inplace([](int& color) { color /= 2; });

  BOOST_CHECK_EQUAL(0, m(0, 0));
  BOOST_CHECK_EQUAL(1, m(1, 0));
  BOOST_CHECK_EQUAL(2, m(0, 1));
  BOOST_CHECK_EQUAL(3, m(1, 1));
}

BOOST_AUTO_TEST_SUITE_END()
