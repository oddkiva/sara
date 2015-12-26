// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Sara/Core/Pixel.hpp>
#include <DO/Sara/Core/Image/Image.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestImageClass, test_2d_image_constructor)
{
  Image<int> image{ 10, 20 };
  EXPECT_EQ(image.width(), 10);
  EXPECT_EQ(image.height(), 20);

  Image<int, 3> volume{ 5, 10, 20 };
  EXPECT_EQ(volume.width(), 5);
  EXPECT_EQ(volume.height(), 10);
  EXPECT_EQ(volume.depth(), 20);

  Image<int, 3> volume2{ volume };
  EXPECT_EQ(volume2.width(), 5);
  EXPECT_EQ(volume2.height(), 10);
  EXPECT_EQ(volume2.depth(), 20);
}

TEST(TestImageClass, test_matrix_view)
{
  Image<int> A{ 2, 3 };
  A.matrix() <<
    1, 2,
    3, 4,
    5, 6;

  EXPECT_EQ(A(0, 0), 1);
  EXPECT_EQ(A(1, 0), 2);
  EXPECT_EQ(A(0, 1), 3);
  EXPECT_EQ(A(1, 1), 4);
  EXPECT_EQ(A(0, 2), 5);
  EXPECT_EQ(A(1, 2), 6);
}

TEST(TestImageClass, test_pixelwise_transform)
{
  auto image = Image<Rgb8>{ 2, 2 };
  image(0, 0) = Rgb8{ 255, 0, 0 }; image(1, 0) = Rgb8{ 255, 0, 0 };
  image(0, 1) = Rgb8{ 0, 255, 0 }; image(1, 1) = Rgb8{ 0, 255, 0 };

  auto result = Image<Rgb64f>{
    image.pixelwise_transform([](const Rgb8& color)
    {
      Rgb64f color_64f;
      smart_convert_color(color, color_64f);
      color_64f = color_64f.cwiseProduct(color_64f);
      return color_64f;
    })
  };
  EXPECT_MATRIX_NEAR(result(0, 0), Rgb64f(1., 0., 0.), 1e-8);
  EXPECT_MATRIX_NEAR(result(1, 0), Rgb64f(1., 0., 0.), 1e-8);
  EXPECT_MATRIX_NEAR(result(0, 1), Rgb64f(0., 1., 0.), 1e-8);
  EXPECT_MATRIX_NEAR(result(1, 1), Rgb64f(0., 1., 0.), 1e-8);
}

TEST(TestImageClass, test_pixelwise_transform_inplace)
{
  Image<Rgb8> image{ 2, 2 };
  image(0, 0) = Rgb8{ 255, 0, 0 }; image(1, 0) = Rgb8{ 255, 0, 0 };
  image(0, 1) = Rgb8{ 0, 255, 0 }; image(1, 1) = Rgb8{ 0, 255, 0 };

  image.pixelwise_transform_inplace([](Rgb8& color) {
    color /= 2;
  });
  EXPECT_MATRIX_EQ(image(0, 0), Rgb8(127, 0, 0));
  EXPECT_MATRIX_EQ(image(1, 0), Rgb8(127, 0, 0));
  EXPECT_MATRIX_EQ(image(0, 1), Rgb8(0, 127, 0));
  EXPECT_MATRIX_EQ(image(1, 1), Rgb8(0, 127, 0));
}

// ========================================================================== //
// Run the tests.
int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
