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
#include <iostream>
#include <vector>
// Google Test.
#include <gtest/gtest.h>
// DO-CV.
#include <DO/Core/Stringify.hpp>
#include <DO/Graphics.hpp>
#include <DO/Graphics/GraphicsUtilities.hpp>
// Local class.
#include "event_scheduler.hpp"

using namespace DO;
using namespace std;

QPointer<EventScheduler> scheduler;

class TestKeyboardMouseInputOnSingleWindow: public testing::Test
{
protected:
  Window test_window_;

  TestKeyboardMouseInputOnSingleWindow()
  {
    test_window_ = openWindow(300, 300);
    scheduler->set_receiver(test_window_);
  }

  virtual ~TestKeyboardMouseInputOnSingleWindow()
  {
    closeWindow(test_window_);
  }
};

TEST_F(TestKeyboardMouseInputOnSingleWindow, test_getMouse)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent event(
    QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
    expected_button, expected_button, Qt::NoModifier
  );
  emit getUserThread().sendEvent(&event, 10);

  int actual_x, actual_y;
  int actual_button = getMouse(actual_x, actual_y);

  EXPECT_EQ(actual_button, 1);
  EXPECT_EQ(actual_x, expected_x);
  EXPECT_EQ(actual_y, expected_y);
}

TEST_F(TestKeyboardMouseInputOnSingleWindow, test_click)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent event(
    QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
    expected_button, expected_button, Qt::NoModifier
  );
  emit getUserThread().sendEvent(&event, 10);

  click();
}

TEST_F(TestKeyboardMouseInputOnSingleWindow, test_getKey)
{
  int expected_key = Qt::Key_A;
  QKeyEvent event(
    QEvent::KeyPress, expected_key, Qt::NoModifier
  );
  emit getUserThread().sendEvent(&event, 10);

  int actual_key = getKey();

  EXPECT_EQ(actual_key, expected_key);
}

TEST_F(TestKeyboardMouseInputOnSingleWindow, test_getEvent_with_no_input_event)
{
  Event event;
  getEvent(50, event);
  EXPECT_EQ(event.type, NO_EVENT);
}

TEST_F(TestKeyboardMouseInputOnSingleWindow, test_getEvent_with_input_key_event)
{
  int expected_key = Qt::Key_A;
  QKeyEvent qt_event(
    QEvent::KeyPress, expected_key, Qt::NoModifier
  );
  emit getUserThread().sendEvent(&qt_event, 5);

  Event event;
  getEvent(50, event);

  EXPECT_EQ(event.type, KEY_PRESSED);
  EXPECT_EQ(event.key, expected_key);
  EXPECT_EQ(event.keyModifiers, Qt::NoModifier);
}

TEST_F(TestKeyboardMouseInputOnSingleWindow, test_getEvent_with_input_mouse_event)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent qt_event(
    QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
    expected_button, expected_button, Qt::NoModifier
  );
  emit getUserThread().sendEvent(&qt_event, 5);

  Event event;
  getEvent(50, event);

  EXPECT_EQ(event.type, MOUSE_RELEASED);
  EXPECT_EQ(event.buttons, 1);
  EXPECT_EQ(event.keyModifiers, Qt::NoModifier);
}

class TestKeyboardMouseInputOnAnyWindow: public testing::Test
{
protected:
  vector<Window> test_windows_;

  TestKeyboardMouseInputOnAnyWindow()
  {
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        test_windows_.push_back(
          openWindow(200, 200, toString(3*i+j), 300*j+300, 300*i+50)
        );
  }

  virtual ~TestKeyboardMouseInputOnAnyWindow()
  {
  }
};

TEST_F(TestKeyboardMouseInputOnAnyWindow, test_anyGetMouse)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;

  for (auto wi : test_windows_)
  {
    setActiveWindow(wi);

    for (auto wj : test_windows_)
    {
      scheduler->set_receiver(wj);
      QMouseEvent event(
        QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
        expected_button, expected_button, Qt::NoModifier
      );
      emit getUserThread().sendEvent(&event, 10);

      Point2i actual_position;
      int actual_button = anyGetMouse(actual_position);

      EXPECT_EQ(actual_button, 1);
      EXPECT_EQ(actual_position.x(), expected_x);
      EXPECT_EQ(actual_position.y(), expected_y);
    }
  }
}

TEST_F(TestKeyboardMouseInputOnAnyWindow, test_anyClick)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;

  for (auto wi : test_windows_)
  {
    setActiveWindow(wi);

    for (auto wj : test_windows_)
    {
      scheduler->set_receiver(wj);
      QMouseEvent event(
        QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
        expected_button, expected_button, Qt::NoModifier
      );
      emit getUserThread().sendEvent(&event, 10);

      anyClick();
    }
  }
}

TEST_F(TestKeyboardMouseInputOnAnyWindow, test_anyGetKey)
{
  int expected_key = Qt::Key_A;

  for (auto wi : test_windows_)
  {
    setActiveWindow(wi);

    for (auto wj : test_windows_)
    {
      scheduler->set_receiver(wj);
      QKeyEvent event(
        QEvent::KeyPress, expected_key, Qt::NoModifier
        );
      emit getUserThread().sendEvent(&event, 10);

      int actual_key = anyGetKey();

      EXPECT_EQ(actual_key, expected_key);
    }
  }
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
  scheduler = new EventScheduler;
  // Connect the user thread and the event scheduler.
  QObject::connect(&getUserThread(), SIGNAL(sendEvent(QEvent *, int)),
                   scheduler.data(), SLOT(schedule_event(QEvent*, int)));

  // Run the worker thread 
  gui_app_.registerUserMain(worker_thread_task);
  int return_code = gui_app_.exec();

  // Cleanup and terminate
  scheduler.clear();
  return return_code;
}