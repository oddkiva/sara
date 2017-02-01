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

#include <DO/Sara/Core.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageIO/Details/Exif.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestImageIO, test_imread_fails)
{
  const auto blank_filepath = string{ "" };
  auto blank_image = Image<Rgb8>{};
  EXPECT_FALSE(imread(blank_image, blank_filepath));
  EXPECT_MATRIX_EQ(blank_image.sizes(), Vector2i::Zero());
}

TEST(TestImageIO, test_rgb_image_read_write)
{
  const string filepaths[] =
  {
    "image.jpg",
    "image.png",
    "image.tif"
  };

  auto true_image = Image<Rgb8>{ 2, 2 };
  true_image(0,0) = White8; true_image(1,0) = Black8;
  true_image(0,1) = Black8; true_image(1,1) = White8;

  for (int i = 0; i < 3; ++i)
  {
    imwrite(true_image, filepaths[i], 100);

    auto image = Image<Rgb8>{};
    EXPECT_TRUE(imread(image, filepaths[i]));
    EXPECT_MATRIX_EQ(image.sizes(), Vector2i(2, 2));

    for (int y = 0; y < true_image.width(); ++y)
      for (int x = 0; x < true_image.height(); ++x)
        EXPECT_MATRIX_EQ(true_image(x, y), image(x, y));
  }
}

TEST(TestImageIO, test_grayscale_image_read_write)
{
  typedef unsigned char gray8u_t;
  auto true_image = Image<gray8u_t>{ 2, 2 };
  true_image.matrix() <<
    255, 0,
    0, 255;

  auto filepath = string{ "image.jpg" };
  auto image = Image<unsigned char>{};
  EXPECT_TRUE(imread(image, filepath));
  EXPECT_MATRIX_EQ(image.sizes(), Vector2i(2, 2));

  for (int y = 0; y < true_image.width(); ++y)
    for (int x = 0; x < true_image.height(); ++x)
      EXPECT_MATRIX_EQ(true_image(x, y), image(x, y));
}

TEST(TestImageIO, test_read_exif_info)
{
  auto filepath = string{ "image.jpg" };
  auto exif_info = EXIFInfo{};
  read_exif_info(exif_info, filepath);

  ostringstream os;
  os << exif_info;
  auto content = os.str();

  auto pos = size_t{};

  pos = content.find("Camera make");        EXPECT_NE(string::npos, pos);
  pos = content.find("Camera model");       EXPECT_NE(string::npos, pos);
  pos = content.find("Software");           EXPECT_NE(string::npos, pos);
  pos = content.find("Bits per sample");    EXPECT_NE(string::npos, pos);
  pos = content.find("Image width");        EXPECT_NE(string::npos, pos);
  pos = content.find("Image height");       EXPECT_NE(string::npos, pos);
  pos = content.find("Image description");  EXPECT_NE(string::npos, pos);
  pos = content.find("Image orientation");  EXPECT_NE(string::npos, pos);
  pos = content.find("Image copyright");    EXPECT_NE(string::npos, pos);
  pos = content.find("Image date/time");    EXPECT_NE(string::npos, pos);
  pos = content.find("Original date/time"); EXPECT_NE(string::npos, pos);
  pos = content.find("Digitize date/time"); EXPECT_NE(string::npos, pos);
  pos = content.find("Subsecond time");     EXPECT_NE(string::npos, pos);
  pos = content.find("Exposure time");      EXPECT_NE(string::npos, pos);
  pos = content.find("F-stop");             EXPECT_NE(string::npos, pos);
  pos = content.find("ISO speed");          EXPECT_NE(string::npos, pos);
  pos = content.find("Subject distance");   EXPECT_NE(string::npos, pos);
  pos = content.find("Exposure bias");      EXPECT_NE(string::npos, pos);
  pos = content.find("Flash used?");        EXPECT_NE(string::npos, pos);
  pos = content.find("Metering mode");      EXPECT_NE(string::npos, pos);
  pos = content.find("Lens focal length");  EXPECT_NE(string::npos, pos);
  pos = content.find("35mm focal length");  EXPECT_NE(string::npos, pos);
  pos = content.find("GPS Latitude");       EXPECT_NE(string::npos, pos);
  pos = content.find("GPS Longitude");      EXPECT_NE(string::npos, pos);
  pos = content.find("GPS Altitude");       EXPECT_NE(string::npos, pos);
}


class TestImageMakeUprightFromExif : public testing::Test
{
protected:
  Image<int> true_image;

  TestImageMakeUprightFromExif() : testing::Test()
  {
    // Draw an 'F' letter.
    true_image.resize(4, 6);
    true_image.matrix() <<
      1, 1, 1, 1,
      1, 0, 0, 0,
      1, 1, 1, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0;
  }

  virtual ~TestImageMakeUprightFromExif() {}
};

TEST_F(TestImageMakeUprightFromExif, test_with_unspecified_tag)
{
  auto image = true_image;

  static_assert(ExifOrientationTag::Unspecified == 0, "Must be 0");

  make_upright_from_exif(image, ExifOrientationTag::Undefined);
  ASSERT_MATRIX_EQ(true_image.matrix(), image.matrix());
}

TEST_F(TestImageMakeUprightFromExif, test_with_upright_tag)
{
  auto image = true_image;

  static_assert(ExifOrientationTag::Upright == 1, "Must be 1");

  make_upright_from_exif(image, ExifOrientationTag::Upright);
  ASSERT_MATRIX_EQ(true_image.matrix(), image.matrix());
}

TEST_F(TestImageMakeUprightFromExif, test_with_flipped_horizontally_tag)
{
  auto image = Image<int>{4, 6};
  image.matrix() <<
      1, 1, 1, 1,
      0, 0, 0, 1,
      0, 1, 1, 1,
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 0, 0, 1;

  static_assert(ExifOrientationTag::FlippedHorizontally == 2, "Must be 2");
  make_upright_from_exif(image, ExifOrientationTag::FlippedHorizontally);
  ASSERT_MATRIX_EQ(true_image.matrix(), image.matrix());
}

TEST_F(TestImageMakeUprightFromExif, test_rotatedccw_180)
{
  auto image = Image<int>{4, 6};
  image.matrix() <<
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 1, 1, 1,
      0, 0, 0, 1,
      1, 1, 1, 1;

  static_assert(ExifOrientationTag::RotatedCCW_180 == 3, "Must be 3");

  make_upright_from_exif(image, ExifOrientationTag::RotatedCCW_180);
  ASSERT_MATRIX_EQ(true_image.matrix(), image.matrix());
}

TEST_F(TestImageMakeUprightFromExif, test_flip_vertically)
{
  auto image = Image<int>{4, 6};
  image.matrix() <<
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 1, 1, 0,
      1, 0, 0, 0,
      1, 1, 1, 1;

  static_assert(ExifOrientationTag::FlippedVertically == 4, "Must be 4");

  make_upright_from_exif(image, ExifOrientationTag::FlippedVertically);
  ASSERT_MATRIX_EQ(true_image.matrix(), image.matrix());
}

TEST_F(TestImageMakeUprightFromExif, test_transpose)
{
  auto image = Image<int>{6, 4};
  image.matrix() <<
      1, 1, 1, 1, 1, 1,
      1, 0, 1, 0, 0, 0,
      1, 0, 1, 0, 0, 0,
      1, 0, 0, 0, 0, 0;

  static_assert(ExifOrientationTag::Transposed == 5, "Must be 5");

  make_upright_from_exif(image, ExifOrientationTag::Transposed);
  ASSERT_MATRIX_EQ(true_image.matrix(), image.matrix());
}

TEST_F(TestImageMakeUprightFromExif, test_with_rotatedccw_90_tag)
{
  auto image = Image<int>{6, 4};
  image.matrix() <<
    1, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 0, 0,
    1, 1, 1, 1, 1, 1;

  static_assert(ExifOrientationTag::RotatedCCW_90 == 6, "Must be 6");

  make_upright_from_exif(image, ExifOrientationTag::RotatedCCW_90);
  ASSERT_MATRIX_EQ(true_image.matrix(), image.matrix());
}

TEST_F(TestImageMakeUprightFromExif, test_with_transverse_tag)
{
  auto image = Image<int>{6, 4};
  image.matrix() <<
      0, 0, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 1,
      0, 0, 0, 1, 0, 1,
      1, 1, 1, 1, 1, 1;

  static_assert(ExifOrientationTag::Transversed == 7, "Must be 7");

  make_upright_from_exif(image, ExifOrientationTag::Transversed);
  ASSERT_MATRIX_EQ(true_image.matrix(), image.matrix());
}

TEST_F(TestImageMakeUprightFromExif, test_with_rotatedcw_90_tag)
{
  auto image = Image<int>{6, 4};
  image.matrix() <<
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 0, 1,
    0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 1;

  static_assert(ExifOrientationTag::RotatedCW_90 == 8, "Must be 8");

  make_upright_from_exif(image, ExifOrientationTag::RotatedCW_90);
  ASSERT_MATRIX_EQ(true_image.matrix(), image.matrix());
}

TEST_F(TestImageMakeUprightFromExif, test_with_undefined_tag)
{
  auto image = true_image;

  static_assert(ExifOrientationTag::Undefined == 9, "Must be 9");

  make_upright_from_exif(image, ExifOrientationTag::Undefined);
  ASSERT_MATRIX_EQ(true_image.matrix(), image.matrix());
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
