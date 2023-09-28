// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageProcessing/Image Pyramid"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core.hpp>

#include <DO/Shakti/Cuda/ImageProcessing/Differential.hpp>

#include "AssertHelpers.hpp"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;


using namespace std;
using namespace sara;


class TestDifferential
{
protected:
  Image<float> _src_image;

  TestDifferential()
  {
    _src_image.resize(3, 3);
    _src_image.matrix() <<
      1, 2, 3,
      1, 2, 3,
      1, 2, 3;
  }
};

BOOST_FIXTURE_TEST_CASE(test_gradient, TestDifferential)
{
  auto& f = _src_image;

  auto nabla_f = Image<Vector2f>{ _src_image.sizes() };
  shakti::compute_gradient(
    reinterpret_cast<shakti::Vector2f *>(nabla_f.data()),
    f.data(),
    f.sizes().data());

  for (auto y = 0; y < nabla_f.height(); ++y)
  {
    for (auto x = 0; x < nabla_f.width(); ++x)
    {
      Vector2f expected_nabla_f_xy{
        x == 1 ? 1 : 0.5f,
        0
      };
      const auto& nabla_f_xy = nabla_f(x, y);
      BOOST_CHECK_SMALL((expected_nabla_f_xy - nabla_f_xy).norm(), 1e-5f);
    }
  }

  f.matrix() = Matrix3f{ f.matrix().transpose() };
  shakti::compute_gradient(
    reinterpret_cast<shakti::Vector2f *>(nabla_f.data()),
    f.data(),
    f.sizes().data());

  for (auto y = 0; y < nabla_f.height(); ++y)
  {
    for (auto x = 0; x < nabla_f.width(); ++x)
    {
      Vector2f expected_nabla_f_xy{
        0,
        y == 1 ? 1 : 0.5f,
      };
      const auto& nabla_f_xy = nabla_f(x, y);
      BOOST_CHECK_SMALL((expected_nabla_f_xy - nabla_f_xy).norm(), 1e-5f);
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_laplacian, TestDifferential)
{
  auto& f = _src_image;
  f.matrix() <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;

  auto laplacian = Image<float>{ 3, 3 };
  Matrix3f true_laplacian;
  true_laplacian.setZero();

  shakti::compute_laplacian(laplacian.data(), f.data(), f.sizes().data());
  BOOST_CHECK_SMALL((laplacian.matrix() - true_laplacian).norm(), 1e-5f);
}

BOOST_FIXTURE_TEST_CASE(test_laplacian_2, TestDifferential)
{
  /*
    We test the following function:
    f(x,y) = x^2 + y^2
    So the laplacian is a constant function, i.e.: tr(D^2 f) (x,y) = 4
    for any (x,y).
  */

  auto f = Image<float>{ 4, 4 };
  f.matrix() <<
    0,  1,  4,  9,
    1,  2,  5, 10,
    4,  5,  8, 13,
    9, 10, 13, 18;

  auto laplacian_f = Image<float>{ 4, 4 };
  shakti::compute_laplacian(laplacian_f.data(), f.data(), f.sizes().data());

  Matrix2f actual_central_block = laplacian_f.matrix().block<2, 2>(1, 1);
  Matrix2f expected_central_block = 4 * Matrix2f::Ones();
  BOOST_CHECK(expected_central_block == actual_central_block);
}

//BOOST_FIXTURE_TEST_CASE(test_hessian, TestDifferential)
//{
//  auto& f = _src_image;
//  f.matrix() <<
//    1, 1, 1,
//    1, 1, 1,
//    1, 1, 1;
//
//  auto hessian_f = Image<Matrix2f>{};
//  compute_hessian(hessian_f);
//  for (auto y = 0; y < hessian_f.height(); ++y)
//  {
//    for (auto x = 0; x < hessian_f.width(); ++x)
//    {
//      Matrix2f true_hessian;
//      true_hessian.setZero();
//      Matrix2f hessian = hessian_f(x, y);
//      BOOST_CHECK_SMALL(true_hessian, hessian, 1e-5);
//    }
//  }
//}

//BOOST_FIXTURE_TEST_CASE(test_hessian_2, TestDifferential)
//{
//  /*
//    We test the following function:
//    f(x,y) = xy;
//    Thus the Hessian function should be equal to:
//    Hf(x,y) = [0 1
//               1 0]
//    at any point (x,y)
//  */
//
//  Image<float> f(4, 4);
//  f.matrix() <<
//    0, 0, 0, 0,
//    0, 1, 2, 3,
//    0, 2, 4, 6,
//    0, 3, 6, 9;
//
//  //Hessian<float> compute_hessian(f);
//  Image<Matrix2f> Hf = hessian(f);
//
//  Matrix2f expected_hessian;
//  expected_hessian <<
//    0, 1,
//    1, 0;
//  for (int y = 1; y < 3; ++y)
//    for (int x = 1; x < 3; ++x)
//      BOOST_CHECK(expected_hessian == Hf(x, y));
//}
