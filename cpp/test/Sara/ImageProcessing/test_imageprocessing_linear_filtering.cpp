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

#define BOOST_TEST_MODULE "ImageProcessing/Linear Filtering"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Math/UsualFunctions.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestConvolutionFilter)

BOOST_AUTO_TEST_CASE(test_convolve_array)
{
  std::vector<float> signal(10, 1);
  std::vector<float> kernel(3, 1);

  convolve_array(&signal[0], &kernel[0], static_cast<int>(signal.size()) - 2,
                 static_cast<int>(kernel.size()));

  for (size_t i = 0; i != signal.size(); ++i)
  {
    if (i > signal.size() - 3)
      BOOST_CHECK_EQUAL(1, signal[i]);
    else
      BOOST_CHECK_EQUAL(3, signal[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()


class TestFixtureForFilters
{
protected:
  Image<float> _src_image;
  vector<float> _kernel;

public:
  TestFixtureForFilters()
  {
    _src_image.resize(3, 3);
    _src_image.matrix() << 1, 2, 3, 1, 2, 3, 1, 2, 3;

    _kernel.resize(3);
    _kernel[0] = -1. / 2;
    _kernel[1] = 0;
    _kernel[2] = 1. / 2;
  }
};

BOOST_FIXTURE_TEST_SUITE(TestFilters, TestFixtureForFilters)

BOOST_AUTO_TEST_CASE(test_row_based_filter)
{
  MatrixXf true_matrix(3, 3);
  true_matrix << 0.5, 1, 0.5, 0.5, 1, 0.5, 0.5, 1, 0.5;

  auto dst_image = Image<float>{};
  BOOST_CHECK_THROW(apply_row_based_filter(_src_image, dst_image, &_kernel[0],
                                           static_cast<int>(_kernel.size())),
                    domain_error);

  dst_image.resize(_src_image.sizes());
  apply_row_based_filter(_src_image, dst_image, &_kernel[0],
                         static_cast<int>(_kernel.size()));
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());
}

BOOST_AUTO_TEST_CASE(test_column_based_filter)
{
  MatrixXf true_matrix(3, 3);
  true_matrix.setZero();

  auto dst_image = Image<float>{};
  BOOST_CHECK_THROW(apply_column_based_filter(_src_image, dst_image,
                                              &_kernel[0],
                                              static_cast<int>(_kernel.size())),
                    domain_error);

  dst_image.resize(_src_image.sizes());
  apply_column_based_filter(_src_image, dst_image, &_kernel[0],
                            static_cast<int>(_kernel.size()));
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());
}

BOOST_AUTO_TEST_CASE(test_row_derivative)
{
  auto true_matrix = MatrixXf(3, 3);
  true_matrix << 1, 2, 1, 1, 2, 1, 1, 2, 1;

  auto dst_image = Image<float>{};
  BOOST_CHECK_THROW(apply_row_derivative(_src_image, dst_image), domain_error);

  dst_image.resize(_src_image.sizes());
  apply_row_derivative(_src_image, dst_image);
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = row_derivative(_src_image);
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());
}

BOOST_AUTO_TEST_CASE(test_column_derivative)
{
  auto dst_image = Image<float>{};
  auto true_matrix = MatrixXf::Zero(3, 3);

  BOOST_CHECK_THROW(apply_column_derivative(_src_image, dst_image),
                    domain_error);

  dst_image.resize(_src_image.sizes());
  apply_column_derivative(_src_image, dst_image);
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = column_derivative(_src_image);
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());
}

BOOST_AUTO_TEST_CASE(test_gaussian)
{
  // Convolve with Dirac.
  const auto n = _src_image.sizes()[0];
  _src_image.flat_array().fill(0.f);
  _src_image(n / 2, n / 2) = 1.f;

  auto true_matrix = MatrixXf(3, 3);
  true_matrix <<
    exp(-1.0f), exp(-0.5f), exp(-1.0f),
    exp(-0.5f), exp(-0.0f), exp(-0.5f),
    exp(-1.0f), exp(-0.5f), exp(-1.f);
  true_matrix /= true_matrix.sum();

  auto dst_image = Image<float>{};
  BOOST_CHECK_THROW(apply_gaussian_filter(_src_image, dst_image, 1.f, 1.f),
                    domain_error);
  SARA_DEBUG << "OK" << std::endl;

  dst_image.resize(_src_image.sizes());
  apply_gaussian_filter(_src_image, dst_image, 1.f, 1.f);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);

  dst_image.clear();
  dst_image = gaussian(_src_image, 1.f, 1.f);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);

  // Last case.
  _src_image.resize(9,
                    9);  // 2 * 4 * 1 + 1 because of Gaussian truncation factor.
  _src_image.flat_array().fill(0.f);
  _src_image(4, 4) = 1.f;
  true_matrix.resize(9, 9);
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j)
      true_matrix(i, j) = exp(-(square(i - 4.f) + square(j - 4.f)) / 2.f);
  true_matrix /= true_matrix.sum();

  dst_image = _src_image.compute<Gaussian>(1.f);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);
}

BOOST_AUTO_TEST_CASE(test_sobel)
{
  _src_image.flat_array().fill(1);

  auto true_matrix = MatrixXf::Zero(3, 3);

  auto dst_image = Image<float>{};
  BOOST_CHECK_THROW(apply_sobel_filter(_src_image, dst_image), domain_error);

  dst_image.resize(_src_image.sizes());
  apply_sobel_filter(_src_image, dst_image);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);

  dst_image.clear();
  dst_image = sobel(_src_image);
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = _src_image.compute<Sobel>();
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);
}

BOOST_AUTO_TEST_CASE(test_scharr)
{
  _src_image.flat_array().fill(1);

  auto true_matrix = MatrixXf::Zero(3, 3);

  auto dst_image = Image<float>{};
  BOOST_CHECK_THROW(apply_scharr_filter(_src_image, dst_image), domain_error);

  dst_image.resize(_src_image.sizes());
  apply_scharr_filter(_src_image, dst_image);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);

  dst_image.clear();
  dst_image = scharr(_src_image);
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = _src_image.compute<Scharr>();
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);
}

BOOST_AUTO_TEST_CASE(test_prewitt)
{
  _src_image.flat_array().fill(1);

  auto true_matrix = MatrixXf::Zero(3, 3);

  auto dst_image = Image<float>{};
  BOOST_CHECK_THROW(apply_prewitt_filter(_src_image, dst_image), domain_error);

  dst_image.resize(_src_image.sizes());
  apply_prewitt_filter(_src_image, dst_image);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);

  dst_image.clear();
  dst_image = prewitt(_src_image);
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = _src_image.compute<Prewitt>();
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);
}

BOOST_AUTO_TEST_CASE(test_2d_non_separable_filter)
{
  auto width = 3, height = 4;
  _src_image.resize(width, height);
  _src_image.flat_array().fill(1);

  const float kernel_2d[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  auto dst_image = Image<float>{};

  auto true_matrix = MatrixXf(height, width);
  true_matrix.fill(9);

  BOOST_CHECK_THROW(
      apply_2d_non_separable_filter(_src_image, dst_image, kernel_2d, 3, 3),
      domain_error);

  dst_image.resize(width, height);
  apply_2d_non_separable_filter(_src_image, dst_image, kernel_2d, 3, 3);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);
}

BOOST_AUTO_TEST_CASE(test_laplacian)
{
  auto width = 3, height = 4;
  _src_image.resize(width, height);
  _src_image.flat_array().fill(1);

  auto true_matrix = MatrixXf::Zero(height, width);

  auto dst_image = Image<float>{};
  BOOST_CHECK_THROW(apply_laplacian_filter(_src_image, dst_image),
                    domain_error);

  dst_image.resize(width, height);
  apply_laplacian_filter(_src_image, dst_image);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);

  dst_image = laplacian_filter(_src_image);
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());
}

BOOST_AUTO_TEST_CASE(test_roberts_cross)
{
  auto width = 3, height = 4;
  _src_image.resize(width, height);
  _src_image.flat_array().fill(1);

  auto true_matrix = MatrixXf::Zero(height, width);

  auto dst_image = Image<float>{};
  BOOST_CHECK_THROW(apply_roberts_cross_filter(_src_image, dst_image),
                    domain_error);

  dst_image.resize(width, height);
  apply_roberts_cross_filter(_src_image, dst_image);
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);

  dst_image.clear();
  dst_image = roberts_cross(_src_image);
  BOOST_CHECK_EQUAL(true_matrix, dst_image.matrix());

  dst_image.clear();
  dst_image = _src_image.compute<RobertsCross>();
  BOOST_CHECK_SMALL_L2_DISTANCE(true_matrix, dst_image.matrix(), 1e-5f);
}

BOOST_AUTO_TEST_SUITE_END()
