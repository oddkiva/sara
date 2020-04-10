// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Halide Backend/Helpers"

#include <DO/Sara/Core/Tensor.hpp>

#include <boost/test/unit_test.hpp>

#include <drafts/Halide/Helpers.hpp>
#include <drafts/Halide/Utilities.hpp>


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


using namespace sara;


BOOST_AUTO_TEST_CASE(test_transpose)
{
  const auto w = 3;
  const auto h = 2;

  auto src = sara::Image<int>{w, h};
  src.matrix() <<
    0, 1, 2,
    3, 4, 5;

  auto dst = sara::Image<int>{h, w};

  auto src_buffer = halide::as_buffer(src);
  auto dst_buffer = halide::as_buffer(dst);

  auto x = halide::Var{"x"};
  auto y = halide::Var{"y"};

  auto id = halide::identity(src_buffer, x, y);
  auto f = halide::transpose(id, x, y);

  f.realize(dst_buffer);

  auto true_dst_matrix = MatrixXi{h, w};
  true_dst_matrix <<
    0, 3,
    1, 4,
    2, 5;

  BOOST_CHECK(true_dst_matrix == dst.matrix());
}


class TestFilters
{
protected:
  sara::Image<float> _src_image;
  std::vector<float> _kernel;

  TestFilters()
  {
    _src_image.resize(3, 3);
    _src_image.matrix() <<
      1, 2, 3,
      1, 2, 3,
      1, 2, 3;

    _kernel.resize(3);
    _kernel[0] = -1. / 2;
    _kernel[1] = 0;
    _kernel[2] = 1. / 2;
  }
};


BOOST_FIXTURE_TEST_CASE(test_conv_x, TestFilters)
{
  Image<float> dst_image{3, 3};
  Matrix3f true_matrix;
  true_matrix << 0.5, 1, 0.5,
                 0.5, 1, 0.5,
                 0.5, 1, 0.5;

  auto src_buffer = halide::as_buffer(_src_image);
  auto dst_buffer = halide::as_buffer(dst_image);
  auto ker_buffer = halide::as_buffer(_kernel);

  auto x = halide::Var{"x"};
  auto y = halide::Var{"y"};
  auto src_func = halide::identity(src_buffer, x, y);
  auto ker_func = halide::shift(ker_buffer, x, -1);

  auto r = halide::RDom{-1, 3};
  auto conv_x = halide::conv_x(src_func, ker_func, x, y, r);

  conv_x.realize(dst_buffer);

  BOOST_CHECK(true_matrix == dst_image.matrix());
}

BOOST_FIXTURE_TEST_CASE(test_conv_y, TestFilters)
{
  Image<float> dst_image{3, 3};
  Matrix3f true_matrix;
  true_matrix.setZero();

  auto src_buffer = halide::as_buffer(_src_image);
  auto dst_buffer = halide::as_buffer(dst_image);
  auto ker_buffer = halide::as_buffer(_kernel);

  auto x = halide::Var{"x"};
  auto y = halide::Var{"y"};
  auto src_func = halide::BoundaryConditions::repeat_edge(src_buffer);
  auto ker_func = halide::shift(ker_buffer, x, -1);

  auto r = halide::RDom{-1, ker_buffer.dim(0).extent()};
  auto conv_y = halide::conv_y(src_func, ker_func, x, y, r);

  conv_y.realize(dst_buffer);

  BOOST_CHECK(true_matrix == dst_image.matrix());
}

// BOOST_FIXTURE_TEST_CASE(test_x_derivative, TestFilters)
// {
//   Image<float> dst_image{ 3, 3 };
//   MatrixXf true_matrix(3, 3);
//   true_matrix << 1, 2, 1,
//                  1, 2, 1,
//                  1, 2, 1;
//
//   shakti::compute_x_derivative(
//     dst_image.data(), _src_image.data(), _src_image.sizes().data());
//   BOOST_CHECK(true_matrix == dst_image.matrix());
// }
//
// BOOST_FIXTURE_TEST_CASE(test_y_derivative, TestFilters)
// {
//   Image<float> dst_image{ 3, 3 };
//   MatrixXf true_matrix(3, 3);
//   true_matrix.setZero();
//
//   shakti::compute_y_derivative(
//     dst_image.data(), _src_image.data(), _src_image.sizes().data());
//   BOOST_CHECK(true_matrix == dst_image.matrix());
// }
//
// BOOST_FIXTURE_TEST_CASE(test_gaussian, TestFilters)
// {
//   // Convolve with Dirac.
//   const auto n = _src_image.sizes()[0];
//   _src_image.flat_array().fill(0.f);
//   _src_image(n / 2, n / 2) = 1.f;
//
//   MatrixXf true_matrix(3, 3);
//   true_matrix <<
//     exp(-1.0f), exp(-0.5f), exp(-1.0f),
//     exp(-0.5f), exp(-0.0f), exp(-0.5f),
//     exp(-1.0f), exp(-0.5f), exp(-1.f);
//   true_matrix /= true_matrix.sum();
//
//   auto dst_image = Image<float>{ _src_image.sizes() };
//
//   auto apply_gaussian_filter = shakti::GaussianFilter{ 1.f, 1 };
//   apply_gaussian_filter(
//     dst_image.data(), _src_image.data(), _src_image.sizes().data());
//   BOOST_CHECK_SMALL((true_matrix - dst_image.matrix()).norm(), 1e-5f);
// }
