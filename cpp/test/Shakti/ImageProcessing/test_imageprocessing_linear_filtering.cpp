// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Shakti/ImageProcessing/Linear Filtering"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Shakti/ImageProcessing.hpp>

#include "../AssertHelpers.hpp"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;

using namespace std;
using namespace sara;


class TestFilters
{
protected:
  Image<float> _src_image;
  vector<float> _kernel;

  TestFilters()
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


BOOST_FIXTURE_TEST_CASE(test_column_based_convolution, TestFilters)
{
  Image<float> dst_image{ 3, 3 };
  MatrixXf true_matrix(3, 3);
  true_matrix << 0.5, 1, 0.5,
                 0.5, 1, 0.5,
                 0.5, 1, 0.5;

  shakti::apply_column_based_convolution(
    dst_image.data(), _src_image.data(), _kernel.data(),
    static_cast<int>(_kernel.size()), _src_image.sizes().data());
  BOOST_CHECK(true_matrix == dst_image.matrix());
}

BOOST_FIXTURE_TEST_CASE(test_row_based_convolution, TestFilters)
{
  Image<float> dst_image{ 3, 3 };
  MatrixXf true_matrix(3, 3);
  true_matrix.setZero();

  shakti::apply_row_based_convolution(
    dst_image.data(), _src_image.data(), _kernel.data(),
    static_cast<int>(_kernel.size()), _src_image.sizes().data());
  BOOST_CHECK(true_matrix == dst_image.matrix());
}

BOOST_FIXTURE_TEST_CASE(test_x_derivative, TestFilters)
{
  Image<float> dst_image{ 3, 3 };
  MatrixXf true_matrix(3, 3);
  true_matrix << 1, 2, 1,
                 1, 2, 1,
                 1, 2, 1;

  shakti::compute_x_derivative(
    dst_image.data(), _src_image.data(), _src_image.sizes().data());
  BOOST_CHECK(true_matrix == dst_image.matrix());
}

BOOST_FIXTURE_TEST_CASE(test_y_derivative, TestFilters)
{
  Image<float> dst_image{ 3, 3 };
  MatrixXf true_matrix(3, 3);
  true_matrix.setZero();

  shakti::compute_y_derivative(
    dst_image.data(), _src_image.data(), _src_image.sizes().data());
  BOOST_CHECK(true_matrix == dst_image.matrix());
}

BOOST_FIXTURE_TEST_CASE(test_gaussian, TestFilters)
{
  // Convolve with Dirac.
  const auto n = _src_image.sizes()[0];
  _src_image.flat_array().fill(0.f);
  _src_image(n / 2, n / 2) = 1.f;

  MatrixXf true_matrix(3, 3);
  true_matrix <<
    exp(-1.0f), exp(-0.5f), exp(-1.0f),
    exp(-0.5f), exp(-0.0f), exp(-0.5f),
    exp(-1.0f), exp(-0.5f), exp(-1.f);
  true_matrix /= true_matrix.sum();

  auto dst_image = Image<float>{ _src_image.sizes() };

  auto apply_gaussian_filter = shakti::GaussianFilter{ 1.f, 1 };
  apply_gaussian_filter(
    dst_image.data(), _src_image.data(), _src_image.sizes().data());
  BOOST_CHECK_SMALL((true_matrix - dst_image.matrix()).norm(), 1e-5f);
}
