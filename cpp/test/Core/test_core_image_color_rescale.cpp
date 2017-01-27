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

#include <stdint.h>

#include <gtest/gtest.h>

#include <DO/Sara/Core/Image/Operations.hpp>
#include <DO/Sara/Core/Pixel/Typedefs.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestImageConversion, test_find_min_max_for_1d_pixel)
{
  auto image = Image<int>{ 10, 20 };

  for (auto y = 0; y < 20; ++y)
    for (auto x = 0; x < 10; ++x)
      image(x, y) = x+y;

  auto min = int{};
  auto max = int{};
  tie(min, max) = find_min_max(image);
  EXPECT_EQ(0, min);
  EXPECT_EQ(9+19, max);
}


TEST(TestImageConversion, test_find_min_max_for_3d_pixel)
{
  auto image = Image<Rgb8>(10, 20);

  for (auto y = 0; y < 20; ++y)
    for (auto x = 0; x < 10; ++x)
      image(x, y).fill(x+y);

  auto min = Rgb8{};
  auto max = Rgb8{};
  tie(min, max) = find_min_max(image);
  EXPECT_EQ(Rgb8(0, 0, 0), min);
  EXPECT_EQ(Rgb8(28, 28, 28), max);
}


TEST(TestImageConversion, test_genericity_of_color_conversion)
{
  double src_data[] = {
    0., 0.,
    1., 1.
  };

  const auto src_image = ImageView<double>{ src_data, Vector2i{ 2, 2} };
  auto dst_image = Image<unsigned char>{};

  EXPECT_THROW(convert(src_image, dst_image), std::domain_error);

  dst_image.resize(src_image.sizes());
  EXPECT_NO_THROW(convert(src_image, dst_image));

  auto expected_dst_image = Image<unsigned char>{ Vector2i{ 2, 2 } };
  expected_dst_image.matrix() <<
      0,   0,
    255, 255;

  EXPECT_EQ(expected_dst_image, dst_image);
}


TEST(TestImageConversion, test_smart_image_conversion)
{
  auto rgb8_image = Image<Rgb8>{2, 2};
  for (auto y = 0; y < rgb8_image.height(); ++y)
    for (auto x = 0; x < rgb8_image.width(); ++x)
      rgb8_image(x, y).fill(255);

  auto float_image = rgb8_image.convert<float>();
  for (auto y = 0; y < float_image.height(); ++y)
    for (auto x = 0; x < float_image.width(); ++x)
      EXPECT_EQ(float_image(x, y), 1.f);

  auto int_image = float_image.convert<int>();
  for (auto y = 0; y < int_image.height(); ++y)
    for (auto x = 0; x < int_image.width(); ++x)
      EXPECT_EQ(int_image(x, y), INT_MAX);

  auto rgb32f_image = rgb8_image.convert<Rgb64f>();
  for (auto y = 0; y < rgb32f_image.height(); ++y)
    for (auto x = 0; x < rgb32f_image.width(); ++x)
      EXPECT_MATRIX_NEAR(rgb32f_image(x, y), Vector3d::Ones(), 1e-6f);

  auto rgb64f_image = int_image.convert<Rgb64f>();
  for (auto y = 0; y < rgb64f_image.height(); ++y)
    for (auto x = 0; x < rgb64f_image.width(); ++x)
      EXPECT_MATRIX_NEAR(rgb64f_image(x, y), Vector3d::Ones(), 1e-7);
}


TEST(TestImageConversion, test_image_color_rescale)
{
  auto rgb_image= Image<Rgb32f>{ 10, 10 };
  for (auto y = 0; y < rgb_image.height(); ++y)
    for (auto x = 0; x < rgb_image.width(); ++x)
      rgb_image(x, y).fill(static_cast<float>(x+y));

  rgb_image = color_rescale(rgb_image);
  auto rgb_min = Rgb32f{};
  auto rgb_max = Rgb32f{};
  tie(rgb_min, rgb_max) = find_min_max(rgb_image);
  EXPECT_EQ(rgb_min, Vector3f::Zero());
  EXPECT_EQ(rgb_max, Vector3f::Ones());

  auto float_image = Image<float>{ 10, 10 };
  for (auto y = 0; y < float_image.height(); ++y)
    for (auto x = 0; x < float_image.width(); ++x)
      float_image(x, y) = static_cast<float>(x+y);

  auto rescaled_float_image1 = color_rescale(float_image);
  auto float_min = float{};
  auto float_max = float{};
  tie(float_min, float_max) = find_min_max(rescaled_float_image1);
  EXPECT_EQ(float_min, 0);
  EXPECT_EQ(float_max, 1);

  auto rescaled_float_image2 = float_image.compute<ColorRescale>(1.f, 2.f);
  tie(float_min, float_max) = find_min_max(rescaled_float_image2);
  EXPECT_EQ(float_min, 1.f);
  EXPECT_EQ(float_max, 2.f);
}


// ========================================================================== //
// Run the tests.
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
