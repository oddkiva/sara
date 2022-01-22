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

#define BOOST_TEST_MODULE "ImageProcessing/Interpolation"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Pixel/Typedefs.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestInterpolation)

BOOST_AUTO_TEST_CASE(test_interpolation_with_float_1)
{
  Image<float> f(2, 2);
  f.matrix() <<
    0, 1,
    0, 1;
  double value;

  for (int x = 0; x < 2; ++x)
  {
    for (int y = 0; y < 2; ++y)
    {
      Vector2d p = Vector2i(x, y).cast<double>();
      value = interpolate(f, p);
      BOOST_REQUIRE_SMALL(f(x, y) - value, 1e-7);
    }
  }

  value = interpolate(f, Vector2d(0.5, 0.0));
  BOOST_REQUIRE_SMALL(0.5 - value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.2));
  BOOST_REQUIRE_SMALL(0.5 - value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.1));
  BOOST_REQUIRE_SMALL(0.5 - value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.8));
  BOOST_REQUIRE_SMALL(0.5 - value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 1.));
  BOOST_REQUIRE_SMALL(0.5 - value, 1e-7);

  f.matrix() <<
    0, 0,
    1, 1;
  value = interpolate(f, Vector2d(0.0, 0.5));
  BOOST_REQUIRE_SMALL(0.5 - value, 1e-7);

  value = interpolate(f, Vector2d(0.2, 0.5));
  BOOST_REQUIRE_SMALL(0.5 - value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.5));
  BOOST_REQUIRE_SMALL(0.5 - value, 1e-7);

  value = interpolate(f, Vector2d(0.8, 0.5));
  BOOST_REQUIRE_SMALL(0.5 - value, 1e-7);

  value = interpolate(f, Vector2d(1, 0.5));
  BOOST_REQUIRE_SMALL(0.5 - value, 1e-7);
}

BOOST_AUTO_TEST_CASE(test_interpolation_with_float_2)
{
  Image<float> f(2, 2);
  f.matrix() <<
    0, 1,
    1, 2;
  double value;

  value = interpolate(f, Vector2d(1, 1));
  BOOST_REQUIRE_SMALL(2 - value, 1e-7);
}

BOOST_AUTO_TEST_CASE(test_interpolation_with_vector2d)
{
  Image<Vector2f> f(2, 2);
  f.matrix()(0,0) = Vector2f::Zero(); f.matrix()(0,1) = Vector2f::Ones();
  f.matrix()(1,0) = Vector2f::Zero(); f.matrix()(1,1) = Vector2f::Ones();

  Vector2d value;

  for (int x = 0; x < 2; ++x)
  {
    for (int y = 0; y < 2; ++y)
    {
      Vector2d p = Vector2i(x, y).cast<double>();
      value = interpolate(f, p);
      BOOST_REQUIRE_SMALL_L2_DISTANCE(f(x, y).cast<double>().eval(), value,
                                      1e-7);
    }
  }

  value = interpolate(f, Vector2d(0.5, 0.0));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Vector2d(0.5, 0.5), value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.2));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Vector2d(0.5, 0.5), value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.1));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Vector2d(0.5, 0.5), value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.8));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Vector2d(0.5, 0.5), value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 1.));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Vector2d(0.5, 0.5), value, 1e-7);

  f.matrix()(0,0) = Vector2f::Zero(); f.matrix()(0,1) = Vector2f::Zero();
  f.matrix()(1,0) = Vector2f::Ones(); f.matrix()(1,1) = Vector2f::Ones();
  value = interpolate(f, Vector2d(0.2, 0.5));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Vector2d(0.5, 0.5), value, 1e-7);
}

BOOST_AUTO_TEST_CASE(test_interpolation_with_rgb64f)
{
  Image<Rgb64f> f(2, 2);
  f.matrix()(0,0) = Rgb64f::Zero(); f.matrix()(0,1) = Rgb64f::Ones();
  f.matrix()(1,0) = Rgb64f::Zero(); f.matrix()(1,1) = Rgb64f::Ones();

  Rgb64f value;

  for (int x = 0; x < 2; ++x)
  {
    for (int y = 0; y < 2; ++y)
    {
      Vector2d p = Vector2i(x, y).cast<double>();
      value = interpolate(f, p);
      BOOST_REQUIRE_SMALL_L2_DISTANCE(f(x, y).cast<double>().eval(), value,
                                      1e-7);
    }
  }

  value = interpolate(f, Vector2d(0.5, 0.0));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Rgb64f(0.5, 0.5, 0.5), value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.2));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Rgb64f(0.5, 0.5, 0.5), value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.1));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Rgb64f(0.5, 0.5, 0.5), value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 0.8));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Rgb64f(0.5, 0.5, 0.5), value, 1e-7);

  value = interpolate(f, Vector2d(0.5, 1.));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Rgb64f(0.5, 0.5, 0.5), value, 1e-7);

  f.matrix()(0,0) = Rgb64f::Zero(); f.matrix()(0,1) = Rgb64f::Zero();
  f.matrix()(1,0) = Rgb64f::Ones(); f.matrix()(1,1) = Rgb64f::Ones();
  value = interpolate(f, Vector2d(0.2, 0.5));
  BOOST_REQUIRE_SMALL_L2_DISTANCE(Rgb64f(0.5, 0.5, 0.5), value, 1e-7);
}

BOOST_AUTO_TEST_SUITE_END()
