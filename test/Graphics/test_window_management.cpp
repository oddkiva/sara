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
#include "event_scheduler.hpp"

using namespace DO;

// N.B.: this is not needed but for very **STRANGELY**, if we don't add this,
// the program seg-faults in Travis CI. We need to sort this out because this
// issue is completely both unfathomable and ridiculous.
EventScheduler *global_scheduler;

TEST(TestWindow, test_open_and_close_window)
{
  Window w = openWindow(300, 300, "My Window", 10, 10);
  EXPECT_NE(w, Window(0));

  EXPECT_EQ(getWindowWidth(w), w->width());
  EXPECT_EQ(getWindowHeight(w), w->height());
  EXPECT_EQ(getWindowSizes(w), Vector2i(w->width(), w->height()));

  PaintingWindow *pw = qobject_cast<PaintingWindow *>(w);
  EXPECT_EQ(pw->windowTitle().toStdString(), "My Window");
  
  QPointer<QWidget> guarded_widget(pw->scrollArea());
  EXPECT_EQ(guarded_widget->pos(), QPoint(10, 10));

  closeWindow(w);
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
}

TEST(TestWindow, test_set_active_window)
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

TEST(TestWindow, test_resize_window)
{
  Window w = openWindow(300, 300, "My Window", 10, 10);
  EXPECT_EQ(w, getActiveWindow());
  EXPECT_EQ(getWindowSizes(w), Vector2i(300, 300));

  fillCircle(100, 100, 30, Red8);

  resizeWindow(500, 500);
  EXPECT_EQ(getWindowSizes(w), Vector2i(500, 500));

  fillCircle(100, 100, 30, Red8);
}

int worker_thread_task(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}

#undef main
int main(int argc, char **argv)
{
  // Create Qt Application.
  GraphicsApplication gui_app_(argc, argv);

  // Create an event scheduler on the GUI thread.
  global_scheduler = new EventScheduler;
  // Connect the user thread and the event scheduler.
  QObject::connect(&getUserThread(), SIGNAL(sendEvent(QEvent *, int)),
                   global_scheduler, SLOT(schedule_event(QEvent*, int)));

  // Run the worker thread 
  gui_app_.registerUserMain(worker_thread_task);
  int return_code = gui_app_.exec();

  // Cleanup and terminate.
  delete global_scheduler;
  return return_code;
}