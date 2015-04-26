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

#include <gtest/gtest.h>

#include <DO/Core/MultiArray.hpp>

#include "../AssertHelpers.hpp"


using namespace DO;
using namespace std;


// =============================================================================
// Test for the 2D case.
class TestIterator2D : public testing::Test
{
protected:
  typedef MultiArray<Vector2i, 2, RowMajor> Image;
  Image image;

  TestIterator2D() : testing::Test()
  {
    image.resize(5, 10);
    for (int i = 0; i < image.rows(); ++i)
      for (int j = 0; j < image.cols(); ++j)
        image(i,j) = Vector2i(i,j);
  }

  virtual ~TestIterator2D() {}
};


class TestRangeIterator2D : public TestIterator2D {};

TEST_F(TestRangeIterator2D, test_row_major_prefix_increment)
{
  Image::array_iterator it(image.begin_array());
  for (int i = 0; i < image.rows(); ++i) {
    for (int j = 0; j < image.cols(); ++j) {
      ASSERT_MATRIX_EQ(*it, Vector2i(i,j));
      ASSERT_MATRIX_EQ(it.position(), Vector2i(i,j));
      ASSERT_EQ(
        it.operator->(),
        image.data() + jump(Vector2i(i,j), image.strides()));
      ++it;
    }
  }
  EXPECT_TRUE(it.end());
}

TEST_F(TestRangeIterator2D, test_row_major_postfix_increment)
{
  Image::array_iterator it(image.begin_array());
  for (int i = 0; i < image.rows(); ++i)
    for (int j = 0; j < image.cols(); ++j)
      ASSERT_MATRIX_EQ(*(it++), Vector2i(i,j));
}

TEST_F(TestRangeIterator2D, test_row_major_prefix_decrement)
{
  Image::array_iterator it(image.begin_array());
  it += image.sizes() - Vector2i::Ones();

  for (int i = image.rows()-1; i >= 0; --i) {
    for (int j = image.cols()-1; j >= 0; --j) {
      ASSERT_MATRIX_EQ(*it, Vector2i(i,j));
      ASSERT_MATRIX_EQ(it.position(), Vector2i(i,j));
      ASSERT_EQ(
        it.operator->(),
        image.data() + jump(Vector2i(i,j), image.strides()));
      --it;
    }
  }
  EXPECT_TRUE(it.end());
}

TEST_F(TestRangeIterator2D, test_row_major_postfix_decrement)
{
  Image::array_iterator it(image.begin_array());
  it += image.sizes() - Vector2i::Ones();

  for (int i = image.rows()-1; i >= 0; --i)
    for (int j = image.cols()-1; j >= 0; --j)
      ASSERT_MATRIX_EQ(*(it--), Vector2i(i,j));
}

TEST_F(TestRangeIterator2D, test_special_dereferencing_operator)
{
  Image::array_iterator it(image.begin_array());
  ASSERT_MATRIX_EQ(it(Vector2i(1,2)), Vector2i(1,2));
  ASSERT_MATRIX_EQ(it(1,2), Vector2i(1,2));
}

TEST_F(TestRangeIterator2D, test_const_multiarray_iterator)
{
  const Image& const_image = image;
  Image::const_array_iterator it = const_image.begin_array();
  while (!it.end())
    ++it;
}

TEST_F(TestRangeIterator2D, test_equality_and_inequality_comparisons)
{
  const Image& const_image = image;

  // Equalities.
  EXPECT_EQ(image.begin_array(), const_image.begin_array());
  EXPECT_EQ(image.begin_array(), image.begin_array()++);
  EXPECT_EQ(image.begin_array(), image.begin());
  EXPECT_EQ(image.begin_array(), image.data());

  EXPECT_MATRIX_EQ(*image.begin_array(), *const_image.begin_array());

  // Inequalities.
  EXPECT_NE(image.begin_array(), ++image.begin_array());
  EXPECT_NE(image.begin_array(), image.data()+1);

  EXPECT_NE(image.begin_array(), ++const_image.begin_array());
  EXPECT_NE(image.begin_array(), const_image.begin()+1);
  EXPECT_NE(image.begin_array(), const_image.data()+1);
}


// =============================================================================
// Test on subarray iterator in 2D.
class TestSubrangeIterator2D : public TestIterator2D
{
protected:
  Vector2i start;
  Vector2i end;

  TestSubrangeIterator2D() : TestIterator2D()
  {
    start << 1, 2;
    end << 4, 8;
  }

};

TEST_F(TestSubrangeIterator2D, test_row_major_prefix_increment)
{
  Image::subarray_iterator it(image.begin_subarray(start, end));
  for (int i = start(0); i < end(0); ++i) {
    for (int j = start(1); j < end(1); ++j) {
      ASSERT_MATRIX_EQ(*it, Vector2i(i,j));
      ASSERT_MATRIX_EQ(it.position(), Vector2i(i,j));
      ASSERT_EQ(
        it.operator->(),
        image.data() + jump(Vector2i(i,j), image.strides()));
      ++it;
    }
  }
  EXPECT_TRUE(it.end());
}

TEST_F(TestSubrangeIterator2D, test_row_major_postfix_increment)
{
  Image::subarray_iterator it(image.begin_subarray(start, end));
  for (int i = start(0); i < end(0); ++i)
    for (int j = start(1); j < end(1); ++j)
      ASSERT_MATRIX_EQ(*(it++), Vector2i(i,j));
}

TEST_F(TestSubrangeIterator2D, test_row_major_prefix_decrement)
{
  Image::subarray_iterator it(image.begin_subarray(start, end));
  it += end - start - Vector2i::Ones();

  for (int i = end(0)-1; i >= start(0); --i) {
    for (int j = end(1)-1; j >= start(1); --j) {
      ASSERT_MATRIX_EQ(*it, Vector2i(i,j));
      ASSERT_MATRIX_EQ(it.position(), Vector2i(i,j));
      ASSERT_EQ(
        it.operator->(),
        image.data() + jump(Vector2i(i,j), image.strides()));
      --it;
    }
  }
  EXPECT_TRUE(it.end());
}

TEST_F(TestSubrangeIterator2D, test_row_major_postfix_decrement)
{
  Image::subarray_iterator it(image.begin_subarray(start, end));
  it += end - start - Vector2i::Ones();

  for (int i = end(0)-1; i >= start(0); --i)
    for (int j = end(1)-1; j >= start(1); --j)
      ASSERT_MATRIX_EQ(*(it--), Vector2i(i,j));
}

TEST_F(TestSubrangeIterator2D, test_special_dereferencing_operator)
{
  Image::subarray_iterator it(image.begin_subarray(start, end));

  Vector2i relative_position(2, -1);
  Vector2i absolute_position = start + relative_position;

  ASSERT_MATRIX_EQ(it(relative_position), absolute_position);
  ASSERT_MATRIX_EQ(it(relative_position), absolute_position);
}

TEST_F(TestSubrangeIterator2D, test_const_multiarray_iterator)
{
  const Image& const_image = image;
  Image::const_subarray_iterator it = const_image.begin_subarray(start, end);
  while (!it.end())
    ++it;
}

TEST_F(TestSubrangeIterator2D, test_equality_and_inequality_comparisons)
{
  const Image& const_image = image;

  // Equalities.
  EXPECT_EQ(image.begin_subarray(start, end),
            const_image.begin_subarray(start, end));
  EXPECT_EQ(image.begin_subarray(Vector2i::Zero(), image.sizes()),
            image.begin_array());
  EXPECT_EQ(image.begin_subarray(Vector2i::Zero(), image.sizes()),
            image.begin());
  EXPECT_EQ(image.begin_subarray(Vector2i::Zero(), image.sizes()),
            image.data());

  EXPECT_MATRIX_EQ(*image.begin_subarray(Vector2i::Zero(), image.sizes()),
                   *const_image.begin_array());

  // Inequalities.
  EXPECT_NE(image.begin_subarray(Vector2i::Zero(), image.sizes()),
            ++image.begin_array());
  EXPECT_NE(image.begin_subarray(Vector2i::Zero(), image.sizes()),
            image.data()+1);

  EXPECT_NE(image.begin_subarray(Vector2i::Zero(), image.sizes()),
            ++const_image.begin_array());
  EXPECT_NE(image.begin_subarray(Vector2i::Zero(), image.sizes()),
            const_image.begin()+1);
  EXPECT_NE(image.begin_subarray(Vector2i::Zero(), image.sizes()),
            const_image.data()+1);
}

// =============================================================================
// Test on 2D axis iterator.
class TestAxisIterator2D : public TestIterator2D {};

TEST_F(TestAxisIterator2D, test_equality_and_inequality_comparisons)
{
  const Image& const_image = image;
  Image::array_iterator it = image.begin_array();

  // Check equalities.
  EXPECT_EQ(it.x(), image.begin_array());
  EXPECT_EQ(it.x(), image.begin_array()++);
  EXPECT_EQ(it.x(), image.begin());
  EXPECT_EQ(it.x(), image.data());
  EXPECT_EQ(it.x(), it.y());

  EXPECT_EQ(it.x(), const_image.begin_array());
  EXPECT_EQ(it.x(), const_image.begin_array()++);
  EXPECT_EQ(it.x(), const_image.begin());
  EXPECT_EQ(it.x(), const_image.data());
  EXPECT_EQ(it.x(), const_image.begin_array().y());

  EXPECT_MATRIX_EQ(*it.x(), *it.y());

  // Inequalities.
  EXPECT_NE(it.x(), ++image.begin_array());
  EXPECT_NE(it.x(), image.begin()+1);
  EXPECT_NE(it.x(), image.data()+1);

  EXPECT_NE(it.x(), ++const_image.begin_array());
  EXPECT_NE(it.x(), const_image.begin()+1);
  EXPECT_NE(it.x(), const_image.data()+1);
}

TEST_F(TestAxisIterator2D, test_iterations)
{
  Image::array_iterator it = image.begin_array();
  EXPECT_EQ(++it.x(), image.begin()+image.stride(0));
  EXPECT_EQ(--it.x(), image.begin());
}

TEST_F(TestAxisIterator2D, test_arithmetic_operations)
{
  Image::array_iterator it = image.begin_array();

  // Addition.
  it.x()+=2;
  EXPECT_EQ(it, image.begin()+image.stride(0)*2);

  // Subtraction.
  it.x()-=2;
  EXPECT_EQ(it, image.begin());

  // Out of range.
  EXPECT_THROW(it.x()-=1, std::out_of_range);
  EXPECT_THROW(it.x()+=image.rows(), std::out_of_range);
}


// =============================================================================
// Test for the 3D case.
class TestIterators3D : public testing::Test
{
protected:
  typedef MultiArray<Vector3i, 3, RowMajor> Volume;
  Volume volume;

  TestIterators3D()
    : testing::Test()
  {
    volume.resize(2, 5, 7);
    for (int i = 0; i < volume.rows(); ++i)
      for (int j = 0; j < volume.cols(); ++j)
        for (int k = 0; k < volume.depth(); ++k)
          volume(i,j,k) = Vector3i(i,j,k);
  }

  virtual ~TestIterators3D() {}
};

TEST_F(TestIterators3D, test_row_major_increment)
{
  Volume::array_iterator it(volume.begin_array());
  for (int i = 0; i < volume.rows(); ++i) {
    for (int j = 0; j < volume.cols(); ++j) {
      for (int k = 0; k < volume.depth(); ++k) {
        ASSERT_MATRIX_EQ(*it, Vector3i(i,j,k));
        ASSERT_MATRIX_EQ(it.position(), Vector3i(i,j,k));
        ASSERT_EQ(
          it.operator->(),
          volume.data() + jump(Vector3i(i,j,k), volume.strides()));
        ++it;
      }
    }
  }
  EXPECT_TRUE(it.end());
}

TEST_F(TestIterators3D, test_row_major_decrement)
{
  Volume::array_iterator it(volume.begin_array());
  it += volume.sizes() - Vector3i::Ones();

  for (int i = volume.rows()-1; i >= 0; --i) {
    for (int j = volume.cols()-1; j >= 0; --j) {
      for (int k = volume.depth()-1; k >= 0; --k) {
        ASSERT_MATRIX_EQ(*it, Vector3i(i,j,k));
        ASSERT_MATRIX_EQ(it.position(), Vector3i(i,j,k));
        ASSERT_EQ(
          it.operator->(),
          volume.data() + jump(Vector3i(i,j,k), volume.strides()));
        --it;
      }
    }
  }
  EXPECT_TRUE(it.end());
}

TEST_F(TestIterators3D, test_special_dereferencing_operator)
{
  Volume::array_iterator it(volume.begin_array());
  EXPECT_MATRIX_EQ(it(Vector3i(1,2,4)), Vector3i(1,2,4));
  EXPECT_MATRIX_EQ(it(1,2,4), Vector3i(1,2,4));
}

TEST_F(TestIterators3D, test_delta)
{
  Volume::array_iterator it(volume.begin_array());
  const int X = 0;
  const int Y = 1;
  int dx = 1;
  int dy = 1;

  EXPECT_MATRIX_EQ(it.delta(X, dx, Y, dy), Vector3i(0+dx, 0+dy, 0));

  Vector3i value = it.delta<X, Y>(dx, dy);
  EXPECT_MATRIX_EQ(value, Vector3i(0+dx, 0+dy, 0));
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
