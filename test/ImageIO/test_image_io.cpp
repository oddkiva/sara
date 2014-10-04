// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Core.hpp>
#include <DO/ImageIO.hpp>

#include "../AssertHelpers.hpp"


using namespace DO;
using namespace std;


TEST(TestImageIO, test_image_reading)
{
  string filepaths[] =
  {
    srcPath("image.jpg"),
    srcPath("image.png"),
    srcPath("image.tif")
  };

  Image<Rgb8> true_image(2, 2);
  true_image(0,0) = White8; true_image(1,0) = Black8;
  true_image(0,1) = Black8; true_image(1,1) = White8;

  for (int i = 0; i < 3; ++i)
  {
    Image<Rgb8> image;

    EXPECT_TRUE(imread(image, filepaths[i]));
    EXPECT_MATRIX_EQ(image.sizes(), Vector2i(2, 2));

    for (int y = 0; y < true_image.width(); ++y)
      for (int x = 0; x < true_image.width(); ++x)
        EXPECT_MATRIX_EQ(true_image(x, y), image(x, y));
  }

}


TEST(TestImageIO, test_read_exif_info)
{
  string filepath = srcPath("image.jpg");
  EXIFInfo exif_info;
  EXPECT_TRUE(read_exif_info(exif_info, filepath));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}