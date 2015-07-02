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

#include <gtest/gtest.h>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/ImageIO.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestImageIO, test_rgb_image_read_write)
{
  string blank_filepath{ "" };
  Image<Rgb8> blank_image;
  EXPECT_FALSE(imread(blank_image, blank_filepath));
  EXPECT_MATRIX_EQ(blank_image.sizes(), Vector2i::Zero());

  string filepaths[] =
  {
    "image.jpg",
    "image.png",
    "image.tif"
  };

  Image<Rgb8> true_image(2, 2);
  true_image(0,0) = White8; true_image(1,0) = Black8;
  true_image(0,1) = Black8; true_image(1,1) = White8;

  for (int i = 0; i < 3; ++i)
  {
    imwrite(true_image, filepaths[i], 100);

    Image<Rgb8> image;
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
  Image<gray8u_t> true_image(2, 2);
  true_image.matrix() <<
    255, 0,
    0, 255;

  string filepath = "image.jpg";
  Image<unsigned char> image;
  EXPECT_TRUE(imread(image, filepath));
  EXPECT_MATRIX_EQ(image.sizes(), Vector2i(2, 2));

  for (int y = 0; y < true_image.width(); ++y)
    for (int x = 0; x < true_image.height(); ++x)
      EXPECT_MATRIX_EQ(true_image(x, y), image(x, y));
}


TEST(TestImageIO, test_read_exif_info)
{
  string filepath = "image.jpg";
  EXIFInfo exif_info;
  read_exif_info(exif_info, filepath);

  ostringstream os;
  os << exif_info;
  string content = os.str();

  size_t pos;

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

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
