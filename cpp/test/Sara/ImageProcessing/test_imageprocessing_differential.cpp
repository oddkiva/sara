// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for.applyr
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageProcessing/Differential Operators"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/Differential.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


class TestFixtureForDifferentialOperators
{
protected:
  Image<float> _src_image;
  vector<float> _kernel;

public:
  TestFixtureForDifferentialOperators()
  {
    _src_image.resize(3, 3);
    _src_image.matrix() << 1, 2, 3, 1, 2, 3, 1, 2, 3;

    _kernel.resize(3);
    _kernel[0] = -1. / 2;
    _kernel[1] = 0;
    _kernel[2] = 1. / 2;
  }
};

BOOST_FIXTURE_TEST_SUITE(TestDifferentialOperator,
                         TestFixtureForDifferentialOperators)

BOOST_AUTO_TEST_CASE(test_gradient)
{
  auto& f = _src_image;
  Vector2i x{1, 1};

  Vector2f gradf_x = gradient(f, x);
  BOOST_CHECK_CLOSE_L2_DISTANCE(Vector2f(1, 0), gradf_x, 1e-5f);

  auto gradf = gradient(f);
  auto gradf_2 = f.compute<Gradient>();
  for (int y = 0; y < gradf.height(); ++y)
  {
    for (int x = 0; x < gradf.width(); ++x)
    {
      auto true_gradf = Vector2f{};
      true_gradf[0] = x == 1 ? 1 : 0.5f;
      true_gradf[1] = 0;

      const auto& gradf_xy = gradf(x, y);
      const auto& gradf_xy_2 = gradf_2(x, y);

      BOOST_CHECK_CLOSE_L2_DISTANCE(true_gradf, gradf_xy, 1e-5f);
      BOOST_CHECK_CLOSE_L2_DISTANCE(true_gradf, gradf_xy_2, 1e-5f);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_laplacian)
{
  auto& f = _src_image;
  f.matrix() << 1, 1, 1, 1, 1, 1, 1, 1, 1;
  Vector2i x{1, 1};

  auto laplacian_x = laplacian(f, x);
  BOOST_CHECK_SMALL(laplacian_x, 1e-5f);

  auto true_delta_f = MatrixXf{3, 3};
  true_delta_f.setZero();

  auto delta_f = Image<float>{};
  delta_f = laplacian(f);
  BOOST_CHECK_SMALL_L2_DISTANCE(delta_f.matrix(), true_delta_f, 1e-5f);

  delta_f.clear();
  delta_f = _src_image.compute<Laplacian>();
  BOOST_CHECK_SMALL_L2_DISTANCE(delta_f.matrix(), true_delta_f, 1e-5f);
}

BOOST_AUTO_TEST_CASE(test_hessian)
{
  auto& f = _src_image;
  f.matrix() <<
    1, 1, 1,
    1, 1, 1,
    1, 1, 1;
  Vector2i x(1, 1);

  auto H_x = hessian(f, x);
  Matrix2f true_H_x = Matrix2f::Zero();
  BOOST_CHECK_SMALL_L2_DISTANCE(true_H_x, H_x, 1e-5f);

  auto hessian_f = hessian(f);
  auto hessian_f_2 = f.compute<Hessian>();

  for (int y = 0; y < hessian_f.height(); ++y)
  {
    for (int x = 0; x < hessian_f.width(); ++x)
    {
      const auto true_hessian = Matrix2f::Zero();
      const auto& hessian = hessian_f(x, y);
      const auto& hessian_2 = hessian_f_2(x, y);
      BOOST_CHECK_SMALL_L2_DISTANCE(true_hessian, hessian, 1e-5f);
      BOOST_CHECK_SMALL_L2_DISTANCE(true_hessian, hessian_2, 1e-5f);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_laplacian_2)
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

  auto laplacian_f = laplacian(f);
  auto laplacian_f_2 = f.compute<Laplacian>();

  Matrix2f actual_central_block = laplacian_f.matrix().block<2, 2>(1, 1);
  Matrix2f actual_central_block_2 = laplacian_f_2.matrix().block<2, 2>(1, 1);
  Matrix2f expected_central_block = 4 * Matrix2f::Ones();
  BOOST_CHECK_EQUAL(expected_central_block, actual_central_block);
  BOOST_CHECK_EQUAL(expected_central_block, actual_central_block_2);

}

BOOST_AUTO_TEST_CASE(test_hessian_2)
{
  /*
    We test the following function:
    f(x,y) = xy;
    Thus the Hessian function should be equal to:
    Hf(x,y) = [0 1
               1 0]
    at any point (x,y)
  */

  auto f = Image<float>{ 4, 4 };
  f.matrix() <<
    0, 0, 0, 0,
    0, 1, 2, 3,
    0, 2, 4, 6,
    0, 3, 6, 9;

  auto Hf = hessian(f);
  auto Hf_2 = f.compute<Hessian>();

  Matrix2f expected_hessian;
  expected_hessian <<
    0, 1,
    1, 0;
  for (int y = 1; y < 3; ++y)
    for (int x = 1; x < 3; ++x)
    {
      BOOST_REQUIRE_EQUAL(expected_hessian, Hf(x, y));
      BOOST_REQUIRE_EQUAL(expected_hessian, Hf_2(x, y));
    }
}

BOOST_AUTO_TEST_SUITE_END()
