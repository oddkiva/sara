// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

// STL.
#include <vector>
// Google Test.
#include <gtest/gtest.h>
#include <QTest>
// DO-CV.
#include <DO/Graphics.hpp>
#include <DO/Graphics/GraphicsUtilities.hpp>

using namespace DO;

class TestPaintingCommands: public testing::Test
{
protected:
  Window test_window_;

  TestPaintingCommands()
  {
    test_window_ = openWindow(300, 300);
  }

  virtual ~TestPaintingCommands()
  {
    closeWindow(test_window_);
  }
};

TEST_F(TestPaintingCommands, test_drawPoint)
{
  EXPECT_TRUE(drawPoint(10, 10, Black8));
  EXPECT_TRUE(drawPoint(10, 10, Color4ub(255, 0, 0, 0)));
  EXPECT_TRUE(drawPoint(Point2f(10.f, 10.f), Red8));
}

TEST_F(TestPaintingCommands, test_drawCircle)
{
  EXPECT_TRUE(drawCircle(10, 10, 5, Black8, 2));
  EXPECT_TRUE(drawCircle(Point2i(10, 10), 5, Black8, 2));
  EXPECT_TRUE(drawCircle(Point2f(10.f, 10.f), 5.f, Black8, 2));
  EXPECT_TRUE(drawCircle(Point2d(10., 10.), 5., Black8, 2));
}

TEST_F(TestPaintingCommands, test_drawEllipse)
{
  EXPECT_TRUE(drawEllipse(10, 10, 50, 100, Black8, 2));
  EXPECT_TRUE(drawEllipse(Point2f(10.f, 10.f), 5.f, 10.f, 40.f, Black8, 2));
  EXPECT_TRUE(drawEllipse(Point2d(10., 10.), 5., 10., 40., Black8, 2));
}

TEST_F(TestPaintingCommands, test_drawLine)
{
  EXPECT_TRUE(drawLine(10, 10, 50, 100, Black8, 2));
  EXPECT_TRUE(drawLine(Point2i(10, 10), Point2i(50, 100), Black8, 2));
  EXPECT_TRUE(drawLine(Point2f(10, 10), Point2f(50, 100), Black8, 2));
  EXPECT_TRUE(drawLine(Point2d(10, 10), Point2d(50, 100), Black8, 2));
}

TEST_F(TestPaintingCommands, test_drawRect)
{
  EXPECT_TRUE(drawRect(10, 10, 50, 100, Black8, 2));
}

TEST_F(TestPaintingCommands, test_drawPoly)
{
  int x[] = {10, 20};
  int y[] = {10, 20};
  EXPECT_TRUE(drawPoly(x, y, 2, Black8, 2));

  Point2i poly[] = {Point2i(10, 10), Point2i(20, 20)};
  EXPECT_TRUE(drawPoly(poly, 2, Red8, 1));
}

TEST_F(TestPaintingCommands, test_drawString)
{
  EXPECT_TRUE(
    drawString(10, 10, "example string", Red8, 14, 0.5,
    true, false, false));
}

TEST_F(TestPaintingCommands, test_drawArrow)
{
  EXPECT_TRUE(drawArrow(10, 10, 30, 30, Red8));
  EXPECT_TRUE(drawArrow(10, 10, 30, 30, Red8, 1., 1., 1, 2));
  EXPECT_TRUE(drawArrow(Point2f(10, 10), Point2f(30, 30), Red8, 3));
}

TEST_F(TestPaintingCommands, test_fillEllipse)
{
  EXPECT_TRUE(fillEllipse(10, 10, 50, 100, Black8));
  EXPECT_TRUE(fillEllipse(Point2i(10, 10), 5, 10, Black8));
}

TEST_F(TestPaintingCommands, test_fillRect)
{
  EXPECT_TRUE(fillRect(10, 10, 50, 100, Black8));
  EXPECT_TRUE(fillRect(Point2i(10, 10), 50, 100, Black8));
}

TEST_F(TestPaintingCommands, test_fillCircle)
{
  EXPECT_TRUE(fillCircle(10, 10, 2, Red8));
  EXPECT_TRUE(fillCircle(Point2i(10, 10), 2, Red8));
  EXPECT_TRUE(fillCircle(Point2f(10.f, 10.f), 2.f, Red8));
}

TEST_F(TestPaintingCommands, test_fillPoly)
{
  int x[] = {10, 20};
  int y[] = {10, 20};
  EXPECT_TRUE(fillPoly(x, y, 2, Black8));

  Point2i poly[] = {Point2i(10, 10), Point2i(20, 20)};
  EXPECT_TRUE(fillPoly(poly, 2, Red8));
}

TEST_F(TestPaintingCommands, test_putColorImage)
{
  int w = 50, h = 50;
  std::vector<Color3ub> data(w*h, Red8);
  EXPECT_TRUE(putColorImage(10, 10, data.data(), w, h, 2.));
  EXPECT_TRUE(putColorImage(Point2i(10, 10), data.data(), w, h, 2.));
}

TEST_F(TestPaintingCommands, test_putGreyImage)
{
  int w = 50, h = 50;
  std::vector<unsigned char> data(w*h, 0);
  EXPECT_TRUE(putGreyImage(10, 10, data.data(), w, h, 2.));
  EXPECT_TRUE(putGreyImage(Point2i(10, 10), data.data(), w, h, 2.));
}

TEST_F(TestPaintingCommands, test_display)
{
  Image<Color3ub> image(50, 50);
  image.array().fill(Red8);
  EXPECT_TRUE(display(image, 0, 0, 1.4));
  EXPECT_TRUE(display(image, Point2i(0, 10), 1.4));

  Image<Rgb8> rgb_image(50, 50);
  rgb_image.array().fill(Red8);
  EXPECT_TRUE(display(rgb_image, 0, 0, 1.4));
  EXPECT_TRUE(display(rgb_image, Point2i(0, 10), 1.4));
}

TEST_F(TestPaintingCommands, test_clearWindow)
{
  EXPECT_TRUE(clearWindow());
}

TEST_F(TestPaintingCommands, test_setAntialiasing)
{
  EXPECT_TRUE(setAntialiasing());
}

TEST_F(TestPaintingCommands, test_setTransparency)
{
  EXPECT_TRUE(setTransparency());
}

TEST_F(TestPaintingCommands, test_saveScreen)
{
  Image<Color3ub> image(50, 50);
  image.array().fill(Red8);
  EXPECT_TRUE(display(image, 0, 0, 1.4));
  EXPECT_TRUE(saveScreen(getActiveWindow(), "test.png"));
}

template <class ColorType>
class TestTemplateDisplay : public testing::Test
{
protected:
  typedef testing::Test Base;
  Window test_window_;

  TestTemplateDisplay(): Base()
  {
    test_window_ = openWindow(300, 300);
  }

  virtual ~TestTemplateDisplay()
  {
    closeWindow(test_window_);
  }
};

typedef testing::Types<
  unsigned char, unsigned short, unsigned int,
  char, short, int, float, double
> ColorTypes;

TYPED_TEST_CASE_P(TestTemplateDisplay);

TYPED_TEST_P(TestTemplateDisplay, test_display)
{
  Image<TypeParam> image(20, 20);
  image.array().fill(Red8);
  EXPECT_TRUE(display(image, 0, 0, 1.4));
  EXPECT_TRUE(display(image, Point2i(0, 10), 1.4));
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}