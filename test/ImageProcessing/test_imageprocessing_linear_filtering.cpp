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


#include <gtest/gtest.h>

#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


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


TEST_F(TestFilters, test_row_based_filter)
{
  MatrixXf true_matrix(3, 3);
  true_matrix << 0.5, 1, 0.5,
                 0.5, 1, 0.5,
                 0.5, 1, 0.5;

  auto dst_image = Image<float>{};
  EXPECT_THROW(apply_row_based_filter(_src_image, dst_image,
                                      &_kernel[0],
                                      static_cast<int>(_kernel.size())),
               domain_error);

  dst_image.resize(_src_image.sizes());
  apply_row_based_filter(_src_image, dst_image,
                         &_kernel[0],
                         static_cast<int>(_kernel.size()));
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());
}


TEST_F(TestFilters, test_column_based_filter)
{
  MatrixXf true_matrix(3, 3);
  true_matrix.setZero();

  auto dst_image = Image<float>{};
  EXPECT_THROW(apply_column_based_filter(_src_image, dst_image, &_kernel[0],
                                         static_cast<int>(_kernel.size())),
               domain_error);

  dst_image.resize(_src_image.sizes());
  apply_column_based_filter(_src_image, dst_image,
                            &_kernel[0],
                            static_cast<int>(_kernel.size()));
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());
}


TEST_F(TestFilters, test_row_derivative)
{
  auto true_matrix = MatrixXf(3, 3);
  true_matrix << 1, 2, 1,
                 1, 2, 1,
                 1, 2, 1;

  auto dst_image = Image<float>{};
  EXPECT_THROW(apply_row_derivative(_src_image, dst_image),
               domain_error);

  dst_image.resize(_src_image.sizes());
  apply_row_derivative(_src_image, dst_image);
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = row_derivative(_src_image);
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());
}


TEST_F(TestFilters, test_column_derivative)
{
  auto dst_image = Image<float>{};
  auto true_matrix = MatrixXf::Zero(3, 3);

  EXPECT_THROW(apply_column_derivative(_src_image, dst_image),
               domain_error);

  dst_image.resize(_src_image.sizes());
  apply_column_derivative(_src_image, dst_image);
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = column_derivative(_src_image);
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());
}


TEST_F(TestFilters, test_gaussian)
{
  // Convolve with Dirac.
  const auto n = _src_image.sizes()[0];
  _src_image.array().fill(0.f);
  _src_image(n / 2, n / 2) = 1.f;

  auto true_matrix = MatrixXf(3, 3);
  true_matrix <<
    exp(-1.0f), exp(-0.5f), exp(-1.0f),
    exp(-0.5f), exp(-0.0f), exp(-0.5f),
    exp(-1.0f), exp(-0.5f), exp(-1.f);
  true_matrix /= true_matrix.sum();

  auto dst_image = Image<float>{};
  EXPECT_THROW(apply_gaussian_filter(_src_image, dst_image, 1.f, 1.f),
               domain_error);

  dst_image.resize(_src_image.sizes());
  apply_gaussian_filter(_src_image, dst_image, 1.f, 1.f);
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);

  dst_image.clear();
  dst_image = gaussian(_src_image, 1.f, 1.f);
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);

  // Last case.
  _src_image.resize(9, 9); // 2 * 4 * 1 + 1 because of Gaussian truncation factor.
  _src_image.array().fill(0.f);
  _src_image(4, 4) = 1.f;
  true_matrix.resize(9, 9);
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j)
      true_matrix(i, j) = exp(-(pow(i - 4.f, 2) + pow(j - 4.f, 2)) / 2.f);
  true_matrix /= true_matrix.sum();

  dst_image = _src_image.compute<Gaussian>(1.f);
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);
}


TEST_F(TestFilters, test_sobel)
{
  _src_image.array().fill(1);

  auto true_matrix = MatrixXf::Zero(3, 3);

  auto dst_image = Image<float>{};
  EXPECT_THROW(apply_sobel_filter(_src_image, dst_image), domain_error);

  dst_image.resize(_src_image.sizes());
  apply_sobel_filter(_src_image, dst_image);
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);

  dst_image.clear();
  dst_image = sobel(_src_image);
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = _src_image.compute<Sobel>();
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);
}


TEST_F(TestFilters, test_scharr)
{
  _src_image.array().fill(1);

  auto true_matrix = MatrixXf::Zero(3, 3);

  auto dst_image = Image<float>{};
  EXPECT_THROW(apply_scharr_filter(_src_image, dst_image), domain_error);

  dst_image.resize(_src_image.sizes());
  apply_scharr_filter(_src_image, dst_image);
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);

  dst_image.clear();
  dst_image = scharr(_src_image);
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = _src_image.compute<Scharr>();
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);
}


TEST_F(TestFilters, test_prewitt)
{
  _src_image.array().fill(1);

  auto true_matrix = MatrixXf::Zero(3, 3);

  auto dst_image = Image<float>{};
  EXPECT_THROW(apply_prewitt_filter(_src_image, dst_image), domain_error);

  dst_image.resize(_src_image.sizes());
  apply_prewitt_filter(_src_image, dst_image);
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);

  dst_image.clear();
  dst_image = prewitt(_src_image);
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = _src_image.compute<Prewitt>();
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);
}


TEST_F(TestFilters, test_2d_non_separable_filter)
{
  auto width = 3, height = 4;
  _src_image.resize(width, height);
  _src_image.array().fill(1);

  const float kernel_2d[] =
  {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
  };

  auto dst_image = Image<float>{};

  auto true_matrix = MatrixXf(height, width);
  true_matrix.fill(9);

  EXPECT_THROW(
      apply_2d_non_separable_filter(_src_image, dst_image, kernel_2d, 3, 3),
      domain_error);

  dst_image.resize(width, height);
  apply_2d_non_separable_filter(_src_image, dst_image, kernel_2d, 3, 3);
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);
}


TEST_F(TestFilters, test_laplacian)
{
  auto width = 3, height = 4;
  _src_image.resize(width, height);
  _src_image.array().fill(1);

  auto true_matrix = MatrixXf::Zero(height, width);

  auto dst_image = Image<float>{};
  EXPECT_THROW(apply_laplacian_filter(_src_image, dst_image), domain_error);

  dst_image.resize(width, height);
  apply_laplacian_filter(_src_image, dst_image);
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);

  dst_image = laplacian_filter(_src_image);
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());
}


TEST_F(TestFilters, test_roberts_cross)
{
  auto width = 3, height = 4;
  _src_image.resize(width, height);
  _src_image.array().fill(1);

  auto true_matrix = MatrixXf::Zero(height, width);

  auto dst_image = Image<float>{};
  EXPECT_THROW(apply_roberts_cross_filter(_src_image, dst_image), domain_error);

  dst_image.resize(width, height);
  apply_roberts_cross_filter(_src_image, dst_image);
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);

  dst_image.clear();
  dst_image = roberts_cross(_src_image);
  EXPECT_MATRIX_EQ(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = _src_image.compute<RobertsCross>();
  EXPECT_MATRIX_NEAR(true_matrix, dst_image.matrix(), 1e-5);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
