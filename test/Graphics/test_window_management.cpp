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

// Google Test.
#include <gtest/gtest.h>
// DO-CV.
#include <DO/Graphics.hpp>
#include <DO/Graphics/GraphicsUtilities.hpp>

using namespace DO;

TEST(TestWindow, test_open_and_close_window)
{
  Window w = openWindow(300, 300, "My Window", 10, 10);
  EXPECT_NE(w, Window(0));

  EXPECT_EQ(getWindowWidth(w), w->width());
  EXPECT_EQ(getWindowHeight(w), w->height());
  EXPECT_EQ(getWindowSizes(w), Vector2i(w->width(), w->height()));

  PaintingWindow *pw = qobject_cast<PaintingWindow *>(w);
  EXPECT_EQ(pw->windowTitle().toStdString(), "My Window");
  EXPECT_EQ(pw->scrollArea()->pos(), QPoint(10, 10));

  closeWindow(w);
  EXPECT_TRUE(pw->scrollArea()->isHidden());
}

TEST(TestWindow, test_open_and_close_gl_window)
{
  Window w = openGLWindow(300, 300, "My Window", 10, 10);
  EXPECT_NE(w, Window(0));

  EXPECT_EQ(getWindowWidth(w), w->width());
  EXPECT_EQ(getWindowHeight(w), w->height());
  EXPECT_EQ(getWindowSizes(w), Vector2i(w->width(), w->height()));
  EXPECT_EQ(w->windowTitle().toStdString(), "My Window");
  EXPECT_EQ(w->pos(), QPoint(10, 10));

  closeWindow(w);
  EXPECT_TRUE(w->isHidden());
}

TEST(TestWindow, test_open_and_close_graphics_view)
{
  Window w = openGraphicsView(300, 300, "My Window", 10, 10);
  EXPECT_NE(w, Window(0));

  EXPECT_EQ(getWindowWidth(w), w->width());
  EXPECT_EQ(getWindowHeight(w), w->height());
  EXPECT_EQ(getWindowSizes(w), Vector2i(w->width(), w->height()));
  EXPECT_EQ(w->windowTitle().toStdString(), "My Window");
  EXPECT_EQ(w->pos(), QPoint(10, 10));

  closeWindow(w);
  EXPECT_TRUE(w->isHidden());
}

TEST(TestWindow, test_get_and_set_active_window)
{
  Window w1 = openWindow(300, 300, "My Window", 10, 10);
  Window w2 = openGLWindow(300, 300, "My GL Window", 10, 10);
  Window w3 = openGraphicsView(300, 300, "My Graphics View", 10, 10);

  EXPECT_EQ(w1, getActiveWindow());

  setActiveWindow(w2);
  EXPECT_EQ(w2, getActiveWindow());

  setActiveWindow(w3);
  EXPECT_EQ(w3, getActiveWindow());

  // TODO: finish up this test.
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}