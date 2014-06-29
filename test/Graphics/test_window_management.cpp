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

// TODO: we still need to check if windows are closed somehow.

// Google Test.
#include <gtest/gtest.h>
// DO-CV.
#include <DO/Graphics.hpp>
#include <DO/Graphics/GraphicsUtilities.hpp>

using namespace DO;

class TestPaintingWindow: public testing::Test
{
protected:
  Window test_window_;

  TestPaintingWindow()
  {
    std::cout << "Constructing window: " << std::endl;
    test_window_ = openWindow(300, 300, "My Window", 10, 10);
    std::cout << "OK!" << std::endl;
  }

  virtual ~TestPaintingWindow()
  {
    std::cout << "Destroying window: " << std::endl;
    closeWindow(test_window_);
    std::cout << "OK!" << std::endl;
  }
};

TEST_F(TestPaintingWindow, test_window_attributes)
{
  std::cout << "Checking attributes" << std::endl;
  EXPECT_EQ(getWindowWidth(test_window_), test_window_->width());
  std::cout << "Checking attributes" << std::endl;
  EXPECT_EQ(getWindowHeight(test_window_), test_window_->height());
  std::cout << "Checking attributes" << std::endl;
  EXPECT_EQ(getWindowSizes(test_window_),
            Vector2i(test_window_->width(), test_window_->height()));

  PaintingWindow *pw = qobject_cast<PaintingWindow *>(test_window_);

  EXPECT_EQ(pw->windowTitle().toStdString(), "My Window");
  EXPECT_EQ(pw->scrollArea()->pos(), QPoint(10, 10));
}

TEST(TestWindow, DISABLED_test_open_and_close_window)
{
  std::cout << "trying to open window: " << std::endl;
  Window w = openWindow(300, 300, "My Window", 10, 10);
  std::cout << "OK" << std::endl;
  EXPECT_NE(w, Window(0));

  std::cout << "check window attributes: " << std::endl;
  EXPECT_EQ(getWindowWidth(w), w->width());
  EXPECT_EQ(getWindowHeight(w), w->height());
  EXPECT_EQ(getWindowSizes(w), Vector2i(w->width(), w->height()));

  PaintingWindow *pw = qobject_cast<PaintingWindow *>(w);
  EXPECT_EQ(pw->windowTitle().toStdString(), "My Window");
  std::cout << "OK" << std::endl;
  
  QPointer<QWidget> guarded_widget(pw->scrollArea());
  EXPECT_EQ(guarded_widget->pos(), QPoint(10, 10));

  std::cout << "closing window: " << std::endl;
  closeWindow(w);
  std::cout << "OK" << std::endl;
}

TEST(TestWindow, DISABLED_test_open_and_close_gl_window)
{
  Window w = openGLWindow(300, 300, "My Window", 10, 10);
  EXPECT_NE(w, Window(0));

  EXPECT_EQ(getWindowWidth(w), w->width());
  EXPECT_EQ(getWindowHeight(w), w->height());
  EXPECT_EQ(getWindowSizes(w), Vector2i(w->width(), w->height()));
  EXPECT_EQ(w->windowTitle().toStdString(), "My Window");
  EXPECT_EQ(w->pos(), QPoint(10, 10));

  closeWindow(w);
}

TEST(TestWindow, DISABLED_test_open_and_close_graphics_view)
{
  Window w = openGraphicsView(300, 300, "My Window", 10, 10);
  EXPECT_NE(w, Window(0));

  EXPECT_EQ(getWindowWidth(w), w->width());
  EXPECT_EQ(getWindowHeight(w), w->height());
  EXPECT_EQ(getWindowSizes(w), Vector2i(w->width(), w->height()));
  EXPECT_EQ(w->windowTitle().toStdString(), "My Window");
  EXPECT_EQ(w->pos(), QPoint(10, 10));

  closeWindow(w);
}

TEST(TestWindow, DISABLED_test_set_active_window)
{
  Window w1 = openWindow(300, 300, "My Window", 10, 10);
  Window w2 = openGLWindow(300, 300, "My GL Window", 10, 10);
  // TODO: FIXME.
  //Window w3 = openGraphicsView(300, 300, "My Graphics View", 10, 10);

  EXPECT_EQ(w1, getActiveWindow());

  setActiveWindow(w2);
  EXPECT_EQ(w2, getActiveWindow());

  // TODO: FIXME.
  //setActiveWindow(w3);
  //EXPECT_EQ(w3, getActiveWindow());

  closeWindow(w1);
  closeWindow(w2);
  // TODO: FIXME.
  //closeWindow(w3);
}

TEST(TestWindow, DISABLED_test_resize_window)
{
  Window w = openWindow(300, 300, "My Window", 10, 10);
  EXPECT_EQ(w, getActiveWindow());
  EXPECT_EQ(getWindowSizes(w), Vector2i(300, 300));

  fillCircle(100, 100, 30, Red8);

  resizeWindow(500, 500);
  EXPECT_EQ(getWindowSizes(w), Vector2i(500, 500));

  fillCircle(100, 100, 30, Red8);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
