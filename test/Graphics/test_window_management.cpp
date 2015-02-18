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
#include "event_scheduler.hpp"

using namespace DO;

// N.B.: this is not needed but for very **STRANGELY**, if we don't add this,
// the program seg-faults in Travis CI. We need to sort this out because this
// issue is completely both unfathomable and ridiculous.
EventScheduler *global_scheduler;

TEST(TestWindow, test_open_and_close_window)
{
  QPointer<PaintingWindow> window = qobject_cast<PaintingWindow *>(
    create_window(300, 300, "My Window", 10, 10)
  );
  QPointer<QWidget> scroll_area(window->scrollArea());

  // Check window dimensions.
  EXPECT_EQ(get_width(window), window->width());
  EXPECT_EQ(get_height(window), window->height());
  EXPECT_EQ(get_sizes(window),
            Vector2i(window->width(), window->height()));
  // Check window title.
  EXPECT_EQ(scroll_area->windowTitle(), QString("My Window"));
  // Check window position.
  EXPECT_EQ(window->x(), 10);
  EXPECT_EQ(window->y(), 10);

  // Check that the widget gets destroyed when we close the window.
  close_window(window);
  millisleep(10);
  EXPECT_TRUE(scroll_area.isNull());
  EXPECT_TRUE(window.isNull());
}

TEST(TestWindow, test_open_and_close_gl_window)
{
  QPointer<QWidget> window = create_gl_window(300, 300, "My Window", 10, 10);

  EXPECT_EQ(get_width(window), window->width());
  EXPECT_EQ(get_height(window), window->height());
  EXPECT_EQ(get_sizes(window),
            Vector2i(window->width(), window->height()));
  EXPECT_EQ(window->windowTitle(), QString("My Window"));
  EXPECT_EQ(window->pos(), QPoint(10, 10));

  close_window(window);
  millisleep(10);
  EXPECT_TRUE(window.isNull());
}

TEST(TestWindow, test_open_and_close_graphics_view)
{
  QPointer<QWidget> window = create_graphics_view(300, 300, "My Window", 10, 10);

  EXPECT_EQ(get_width(window), window->width());
  EXPECT_EQ(get_height(window), window->height());
  EXPECT_EQ(get_sizes(window),
            Vector2i(window->width(), window->height()));
  EXPECT_EQ(window->windowTitle(), QString("My Window"));
  EXPECT_EQ(window->pos(), QPoint(10, 10));

  close_window(window);
  millisleep(10);
  EXPECT_TRUE(window.isNull());
}

TEST(TestWindow, test_set_active_window)
{
  Window w1 = create_window(300, 300, "My Window", 10, 10);
  Window w2 = create_gl_window(300, 300, "My GL Window", 10, 10);
  Window w3 = create_graphics_view(300, 300, "My Graphics View", 10, 10);

  EXPECT_TRUE(w1);
  EXPECT_TRUE(w2);
  EXPECT_TRUE(w3);

  EXPECT_EQ(w1, active_window());

  set_active_window(w2);
  EXPECT_EQ(w2, active_window());

  set_active_window(w3);
  EXPECT_EQ(w3, active_window());

  close_window(w1);
  close_window(w2);
  close_window(w3);
}

TEST(TestWindow, test_resize_window)
{
  Window w = create_window(300, 300, "My Window", 10, 10);
  EXPECT_EQ(w, active_window());
  EXPECT_EQ(get_sizes(w), Vector2i(300, 300));

  fill_circle(100, 100, 30, Red8);

  resize_window(500, 500);
  EXPECT_EQ(get_sizes(w), Vector2i(500, 500));

  fill_circle(100, 100, 30, Red8);
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
  QObject::connect(&get_user_thread(), SIGNAL(sendEvent(QEvent *, int)),
                   global_scheduler, SLOT(schedule_event(QEvent*, int)));

  // Run the worker thread 
  gui_app_.register_user_main(worker_thread_task);
  int return_code = gui_app_.exec();

  // Cleanup and terminate.
  delete global_scheduler;
  return return_code;
}