// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <vector>

#include <gtest/gtest.h>

#include <QTest>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>

#include "event_scheduler.cpp"


using namespace DO::Sara;


EventScheduler *global_scheduler;


class TestPaintingCommands: public testing::Test
{
protected:
  Window test_window_;

  TestPaintingCommands()
  {
    test_window_ = create_window(300, 300);
  }

  virtual ~TestPaintingCommands()
  {
    close_window(test_window_);
  }
};

TEST_F(TestPaintingCommands, test_draw_point)
{
  EXPECT_TRUE(draw_point(10, 10, Black8));
  EXPECT_TRUE(draw_point(10, 10, Color4ub(255, 0, 0, 0)));
  EXPECT_TRUE(draw_point(Point2f(10.f, 10.f), Red8));
}

TEST_F(TestPaintingCommands, test_draw_circle)
{
  EXPECT_TRUE(draw_circle(10, 10, 5, Black8, 2));
  EXPECT_TRUE(draw_circle(Point2i(10, 10), 5, Black8, 2));
  EXPECT_TRUE(draw_circle(Point2f(10.f, 10.f), 5.f, Black8, 2));
  EXPECT_TRUE(draw_circle(Point2d(10., 10.), 5., Black8, 2));
}

TEST_F(TestPaintingCommands, test_draw_ellipse)
{
  EXPECT_TRUE(draw_ellipse(10, 10, 50, 100, Black8, 2));
  EXPECT_TRUE(draw_ellipse(Point2f(10.f, 10.f), 5.f, 10.f, 40.f, Black8, 2));
  EXPECT_TRUE(draw_ellipse(Point2d(10., 10.), 5., 10., 40., Black8, 2));
}

TEST_F(TestPaintingCommands, test_draw_line)
{
  EXPECT_TRUE(draw_line(10, 10, 50, 100, Black8, 2));
  EXPECT_TRUE(draw_line(Point2i(10, 10), Point2i(50, 100), Black8, 2));
  EXPECT_TRUE(draw_line(Point2f(10, 10), Point2f(50, 100), Black8, 2));
  EXPECT_TRUE(draw_line(Point2d(10, 10), Point2d(50, 100), Black8, 2));
}

TEST_F(TestPaintingCommands, test_draw_rect)
{
  EXPECT_TRUE(draw_rect(10, 10, 50, 100, Black8, 2));
}

TEST_F(TestPaintingCommands, test_draw_poly)
{
  int x[] = {10, 20};
  int y[] = {10, 20};
  EXPECT_TRUE(draw_poly(x, y, 2, Black8, 2));

  Point2i poly[] = {Point2i(10, 10), Point2i(20, 20)};
  EXPECT_TRUE(draw_poly(poly, 2, Red8, 1));
}

TEST_F(TestPaintingCommands, test_draw_string)
{
  EXPECT_TRUE(
    draw_string(10, 10, "example string", Red8, 14, 0.5,
    true, false, false));
}

TEST_F(TestPaintingCommands, test_draw_arrow)
{
  EXPECT_TRUE(draw_arrow(10, 10, 30, 30, Red8));
  EXPECT_TRUE(draw_arrow(10, 10, 30, 30, Red8, 1., 1., 1, 2));
  EXPECT_TRUE(draw_arrow(Point2f(10, 10), Point2f(30, 30), Red8, 3));
}

TEST_F(TestPaintingCommands, test_fill_ellipse)
{
  EXPECT_TRUE(fill_ellipse(10, 10, 50, 100, Black8));
  EXPECT_TRUE(fill_ellipse(Point2i(10, 10), 5, 10, Black8));
}

TEST_F(TestPaintingCommands, test_fill_rect)
{
  EXPECT_TRUE(fill_rect(10, 10, 50, 100, Black8));
  EXPECT_TRUE(fill_rect(Point2i(10, 10), 50, 100, Black8));
}

TEST_F(TestPaintingCommands, test_fill_circle)
{
  EXPECT_TRUE(fill_circle(10, 10, 2, Red8));
  EXPECT_TRUE(fill_circle(Point2i(10, 10), 2, Red8));
  EXPECT_TRUE(fill_circle(Point2f(10.f, 10.f), 2.f, Red8));
}

TEST_F(TestPaintingCommands, test_fill_poly)
{
  int x[] = {10, 20};
  int y[] = {10, 20};
  EXPECT_TRUE(fill_poly(x, y, 2, Black8));

  Point2i poly[] = {Point2i(10, 10), Point2i(20, 20)};
  EXPECT_TRUE(fill_poly(poly, 2, Red8));
}

TEST_F(TestPaintingCommands, test_put_color_image)
{
  int w = 50, h = 50;
  std::vector<Color3ub> data(w*h, Red8);
  EXPECT_TRUE(put_color_image(10, 10, data.data(), w, h, 2.));
  EXPECT_TRUE(put_color_image(Point2i(10, 10), data.data(), w, h, 2.));
}

TEST_F(TestPaintingCommands, test_put_grey_image)
{
  int w = 50, h = 50;
  std::vector<unsigned char> data(w*h, 0);
  EXPECT_TRUE(put_grey_image(10, 10, data.data(), w, h, 2.));
  EXPECT_TRUE(put_grey_image(Point2i(10, 10), data.data(), w, h, 2.));
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

TEST_F(TestPaintingCommands, test_clear_window)
{
  EXPECT_TRUE(clear_window());
}

TEST_F(TestPaintingCommands, test_set_antialiasing)
{
  EXPECT_TRUE(set_antialiasing());
}

TEST_F(TestPaintingCommands, test_set_transparency)
{
  EXPECT_TRUE(set_transparency());
}

TEST_F(TestPaintingCommands, test_save_screen)
{
  Image<Color3ub> image(50, 50);
  image.array().fill(Red8);
  EXPECT_TRUE(display(image, 0, 0, 1.4));
  EXPECT_TRUE(save_screen(active_window(), "test.png"));
}

template <class ColorType>
class TestTemplateDisplay : public testing::Test
{
protected:
  typedef testing::Test Base;
  Window test_window_;

  TestTemplateDisplay(): Base()
  {
    test_window_ = create_window(300, 300);
  }

  virtual ~TestTemplateDisplay()
  {
    close_window(test_window_);
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

int worker_thread(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#undef main
int main(int argc, char **argv)
{
  // Create Qt Application.
  GraphicsApplication gui_app(argc, argv);

  // Create an event scheduler on the GUI thread.
  global_scheduler = new EventScheduler;
  // Connect the user thread and the event scheduler.
  QObject::connect(&get_user_thread(), SIGNAL(sendEvent(QEvent *, int)),
                   global_scheduler, SLOT(schedule_event(QEvent*, int)));

  // Run the worker thread
  gui_app.register_user_main(worker_thread);
  int return_code = gui_app.exec();

  // Cleanup and terminate.
  delete global_scheduler;
  return return_code;
}
