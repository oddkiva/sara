// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Core/Image/Image View Transform Operations"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Image/Operations.hpp>


using namespace std;
using namespace DO::Sara;


class TestFixtureForImageViewTransformOperations
{
protected:
  Image<float> image;
  Vector2i sizes;

public:
  TestFixtureForImageViewTransformOperations()
  {
    sizes << 3, 4;
    image.resize(sizes);
    image.matrix() <<
      1, 1, 1,
      2, 2, 2,
      3, 3, 3,
      4, 4, 4;
  }
};

BOOST_FIXTURE_TEST_SUITE(TestImageViewTransformOperations,
                         TestFixtureForImageViewTransformOperations)

BOOST_AUTO_TEST_CASE(test_with_safe_crop_functor)
{
  const auto a = Vector2i{1, 1};
  const auto b = Vector2i{3, 2};
  auto cropped_image = image.compute<SafeCrop>(a, b);

  auto true_cropped_image = Image<float, 2>{2, 1};
  true_cropped_image.matrix() << 2, 2;
  BOOST_CHECK_EQUAL(true_cropped_image.sizes(), cropped_image.sizes());
  BOOST_CHECK_EQUAL(true_cropped_image.matrix(), cropped_image.matrix());

  const auto x = 1, y = 1;
  const auto w = 2, h = 1;
  cropped_image = safe_crop(image, Point2i{x, y}, Point2i{x + w, y + h});
  BOOST_CHECK_EQUAL(true_cropped_image.sizes(), cropped_image.sizes());
  BOOST_CHECK_EQUAL(true_cropped_image.matrix(), cropped_image.matrix());
}


BOOST_AUTO_TEST_CASE(test_with_safe_crop_lambda)
{
  auto safe_crop_ = [](const Image<float>& src, const Vector2i& a,
                       const Vector2i& b) { return safe_crop(src, a, b); };

  auto a = Vector2i{-3, -3};
  auto b = Vector2i{0, 0};
  auto cropped_image = image.compute(safe_crop_, a, b);

  auto true_cropped_image = Image<float, 2>{3, 3};
  true_cropped_image.matrix().fill(0);
  BOOST_CHECK_EQUAL(true_cropped_image.sizes(), cropped_image.sizes());
  BOOST_CHECK_EQUAL(true_cropped_image.matrix(), cropped_image.matrix());

  auto x = -3, y = -3;
  auto w = 3, h = 3;
  cropped_image = safe_crop(image, Point2i{x, y}, Point2i{x + w, y + h});
  BOOST_CHECK_EQUAL(true_cropped_image.sizes(), cropped_image.sizes());
  BOOST_CHECK_EQUAL(true_cropped_image.matrix(), cropped_image.matrix());

  const auto center = Point2i{-2, -2};
  const auto radius = 1;
  cropped_image = safe_crop(image, center, radius);
  BOOST_CHECK_EQUAL(true_cropped_image.sizes(), cropped_image.sizes());
  BOOST_CHECK_EQUAL(true_cropped_image.matrix(), cropped_image.matrix());
}

BOOST_AUTO_TEST_SUITE_END()
