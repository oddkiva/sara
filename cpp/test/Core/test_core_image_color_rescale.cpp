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

#define BOOST_TEST_MODULE "Core/Image/Color Rescaling"

#include <cstdint>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Image/Operations.hpp>
#include <DO/Sara/Core/Pixel/Typedefs.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestImageConversion)

BOOST_AUTO_TEST_CASE(test_find_min_max_for_1d_pixel)
{
  auto image = Image<int>{10, 20};

  for (auto y = 0; y < 20; ++y)
    for (auto x = 0; x < 10; ++x)
      image(x, y) = x + y;

  auto min = int{};
  auto max = int{};
  tie(min, max) = find_min_max(image);
  BOOST_CHECK_EQUAL(0, min);
  BOOST_CHECK_EQUAL(9 + 19, max);
}

BOOST_AUTO_TEST_CASE(test_find_min_max_for_3d_pixel)
{
  auto image = Image<Rgb8>(10, 20);

  for (auto y = 0; y < 20; ++y)
    for (auto x = 0; x < 10; ++x)
      image(x, y).fill(x + y);

  auto min = Rgb8{};
  auto max = Rgb8{};
  tie(min, max) = find_min_max(image);
  BOOST_CHECK_EQUAL(Rgb8(0, 0, 0), min);
  BOOST_CHECK_EQUAL(Rgb8(28, 28, 28), max);
}

BOOST_AUTO_TEST_CASE(test_genericity_of_color_conversion)
{
  double src_data[] = {0., 0., 1., 1.};

  const auto src_image = ImageView<double>{src_data, Vector2i{2, 2}};
  auto dst_image = Image<unsigned char>{};

  BOOST_CHECK_THROW(convert(src_image, dst_image), std::domain_error);

  dst_image.resize(src_image.sizes());
  BOOST_CHECK_NO_THROW(convert(src_image, dst_image));

  auto expected_dst_image = Image<unsigned char>{Vector2i{2, 2}};
  expected_dst_image.matrix() << 0, 0, 255, 255;

  BOOST_CHECK(expected_dst_image == dst_image);
}

BOOST_AUTO_TEST_CASE(test_smart_image_conversion)
{
  auto rgb8_image = Image<Rgb8>{2, 2};
  for (auto y = 0; y < rgb8_image.height(); ++y)
    for (auto x = 0; x < rgb8_image.width(); ++x)
      rgb8_image(x, y).fill(255);

  auto float_image = rgb8_image.convert<float>();
  for (auto y = 0; y < float_image.height(); ++y)
    for (auto x = 0; x < float_image.width(); ++x)
      BOOST_REQUIRE_EQUAL(float_image(x, y), 1.f);

  auto int_image = float_image.convert<int>();
  for (auto y = 0; y < int_image.height(); ++y)
    for (auto x = 0; x < int_image.width(); ++x)
      BOOST_REQUIRE_EQUAL(int_image(x, y), INT_MAX);

  auto rgb32f_image = rgb8_image.convert<Rgb32f>();
  for (auto y = 0; y < rgb32f_image.height(); ++y)
    for (auto x = 0; x < rgb32f_image.width(); ++x)
      BOOST_REQUIRE_SMALL((rgb32f_image(x, y) - Vector3f::Ones()).norm(),
                          1e-6f);

  auto rgb64f_image = int_image.convert<Rgb64f>();
  for (auto y = 0; y < rgb64f_image.height(); ++y)
    for (auto x = 0; x < rgb64f_image.width(); ++x)
      BOOST_REQUIRE_SMALL((rgb64f_image(x, y) - Vector3d::Ones()).norm(), 1e-7);
}

BOOST_AUTO_TEST_CASE(test_image_color_rescale)
{
  auto rgb_image = Image<Rgb32f>{10, 10};
  for (auto y = 0; y < rgb_image.height(); ++y)
    for (auto x = 0; x < rgb_image.width(); ++x)
      rgb_image(x, y).fill(static_cast<float>(x + y));

  rgb_image = color_rescale(rgb_image);
  auto rgb_min = Rgb32f{};
  auto rgb_max = Rgb32f{};
  tie(rgb_min, rgb_max) = find_min_max(rgb_image);
  BOOST_CHECK_EQUAL(rgb_min, Vector3f::Zero());
  BOOST_CHECK_EQUAL(rgb_max, Vector3f::Ones());

  auto float_image = Image<float>{10, 10};
  for (auto y = 0; y < float_image.height(); ++y)
    for (auto x = 0; x < float_image.width(); ++x)
      float_image(x, y) = static_cast<float>(x + y);

  auto rescaled_float_image1 = color_rescale(float_image);
  auto float_min = float{};
  auto float_max = float{};
  tie(float_min, float_max) = find_min_max(rescaled_float_image1);
  BOOST_CHECK_EQUAL(float_min, 0);
  BOOST_CHECK_EQUAL(float_max, 1);

  auto rescaled_float_image2 = float_image.compute<ColorRescale>(1.f, 2.f);
  tie(float_min, float_max) = find_min_max(rescaled_float_image2);
  BOOST_CHECK_EQUAL(float_min, 1.f);
  BOOST_CHECK_EQUAL(float_max, 2.f);
}

BOOST_AUTO_TEST_SUITE_END()
