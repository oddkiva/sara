// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <vector>

#include <gtest/gtest.h>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>

#include "event_scheduler.hpp"


using namespace DO::Sara;
using namespace std;


EventScheduler *global_scheduler;


class TestKeyboardMouseInputOnSingleWindow: public testing::Test
{
protected:
  Window test_window_;

  TestKeyboardMouseInputOnSingleWindow()
  {
    test_window_ = create_window(300, 300);
    global_scheduler->set_receiver(test_window_);
  }

  virtual ~TestKeyboardMouseInputOnSingleWindow()
  {
    close_window(test_window_);
  }
};

TEST_F(TestKeyboardMouseInputOnSingleWindow,
       test_get_mouse)
{
  Qt::MouseButton expected_qt_mouse_buttons[] = {
    Qt::LeftButton,
    Qt::MiddleButton,
    Qt::RightButton,
  };

  int expected_button_codes[] = {
    MOUSE_LEFT_BUTTON,
    MOUSE_MIDDLE_BUTTON,
    MOUSE_RIGHT_BUTTON
  };

  int expected_x = 150, expected_y = 150;

  for (int i = 0; i < 3; ++i)
  {
    Qt::MouseButton input_qt_mouse_button = expected_qt_mouse_buttons[i];
    int expected_button_code = expected_button_codes[i];

    QMouseEvent event(
      QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
      input_qt_mouse_button, Qt::MouseButtons(input_qt_mouse_button), Qt::NoModifier
      );
    emit get_user_thread().sendEvent(&event, 10);

    int actual_x, actual_y;
    int actual_button = get_mouse(actual_x, actual_y);

    EXPECT_EQ(actual_button, expected_button_code);
    EXPECT_EQ(actual_x, expected_x);
    EXPECT_EQ(actual_y, expected_y);
  }
}

TEST_F(TestKeyboardMouseInputOnSingleWindow,
       test_click)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent event(
    QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
    expected_button, expected_button, Qt::NoModifier
  );
  emit get_user_thread().sendEvent(&event, 10);

  click();
}

TEST_F(TestKeyboardMouseInputOnSingleWindow,
       test_get_key)
{
  int expected_key = Qt::Key_A;
  QKeyEvent event(
    QEvent::KeyPress, expected_key, Qt::NoModifier
  );
  emit get_user_thread().sendEvent(&event, 10);

  int actual_key = get_key();

  EXPECT_EQ(actual_key, expected_key);
}

TEST_F(TestKeyboardMouseInputOnSingleWindow,
       test_get_event_with_no_input_event)
{
  Event event;
  get_event(50, event);
  EXPECT_EQ(event.type, NO_EVENT);
}

TEST_F(TestKeyboardMouseInputOnSingleWindow,
       test_get_event_with_input_key_event)
{
  int expected_key = Qt::Key_A;
  QKeyEvent qt_event(
    QEvent::KeyPress, expected_key, Qt::NoModifier
  );
  emit get_user_thread().sendEvent(&qt_event, 5);

  Event event;
  get_event(50, event);

  EXPECT_EQ(event.type, KEY_PRESSED);
  EXPECT_EQ(event.key, expected_key);
  EXPECT_EQ(event.keyModifiers, static_cast<int>(Qt::NoModifier));
}

TEST_F(TestKeyboardMouseInputOnSingleWindow,
       test_get_event_with_mouse_pressed_event)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent qt_event(
    QEvent::MouseButtonPress, QPointF(expected_x, expected_y),
    expected_button, expected_button, Qt::NoModifier
  );
  emit get_user_thread().sendEvent(&qt_event, 5);

  Event event;
  get_event(50, event);

  EXPECT_EQ(event.type, MOUSE_PRESSED);
  EXPECT_EQ(event.buttons, 1);
  EXPECT_EQ(event.keyModifiers, static_cast<int>(Qt::NoModifier));
}

TEST_F(TestKeyboardMouseInputOnSingleWindow,
       test_get_event_with_mouse_released_event)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent qt_event(
    QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
    expected_button, expected_button, Qt::NoModifier
  );
  emit get_user_thread().sendEvent(&qt_event, 5);

  Event event;
  get_event(50, event);

  EXPECT_EQ(event.type, MOUSE_RELEASED);
  EXPECT_EQ(event.buttons, 1);
  EXPECT_EQ(event.keyModifiers, static_cast<int>(Qt::NoModifier));
}

TEST_F(TestKeyboardMouseInputOnSingleWindow,
       test_get_event_with_mouse_moved_event)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent qt_event(
    QEvent::MouseMove, QPointF(expected_x, expected_y),
    expected_button, expected_button, Qt::NoModifier
  );
  emit get_user_thread().sendEvent(&qt_event, 5);

  Event event;
  get_event(50, event);

  EXPECT_EQ(event.type, MOUSE_RELEASED);
  EXPECT_EQ(event.buttons, 1);
  EXPECT_EQ(event.keyModifiers, static_cast<int>(Qt::NoModifier));
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
          create_window(200, 200, to_string(3*i+j), 300*j+300, 300*i+50)
        );
  }

  virtual ~TestKeyboardMouseInputOnAnyWindow()
  {
  }
};

TEST_F(TestKeyboardMouseInputOnAnyWindow, test_any_get_mouse)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;

  for (auto wi : test_windows_)
  {
    set_active_window(wi);

    for (auto wj : test_windows_)
    {
      global_scheduler->set_receiver(wj);
      QMouseEvent event(
        QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
        expected_button, expected_button, Qt::NoModifier
      );
      emit get_user_thread().sendEvent(&event, 10);

      Point2i actual_position;
      int actual_button = any_get_mouse(actual_position);

      EXPECT_EQ(actual_button, 1);
      EXPECT_EQ(actual_position.x(), expected_x);
      EXPECT_EQ(actual_position.y(), expected_y);
    }
  }
}

TEST_F(TestKeyboardMouseInputOnAnyWindow, test_any_click)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;

  for (auto wi : test_windows_)
  {
    set_active_window(wi);

    for (auto wj : test_windows_)
    {
      global_scheduler->set_receiver(wj);
      QMouseEvent event(
        QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
        expected_button, expected_button, Qt::NoModifier
      );
      emit get_user_thread().sendEvent(&event, 10);

      any_click();
    }
  }
}

TEST_F(TestKeyboardMouseInputOnAnyWindow, test_any_get_key)
{
  int expected_key = Qt::Key_A;

  for (auto wi : test_windows_)
  {
    set_active_window(wi);

    for (auto wj : test_windows_)
    {
      global_scheduler->set_receiver(wj);
      QKeyEvent event(
        QEvent::KeyPress, expected_key, Qt::NoModifier
        );
      emit get_user_thread().sendEvent(&event, 10);

      int actual_key = any_get_key();

      EXPECT_EQ(actual_key, expected_key);
    }
  }
}

int worker_thread(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

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
  gui_app_.register_user_main(worker_thread);
  int return_code = gui_app_.exec();

  // Cleanup and terminate
  delete global_scheduler;
  return return_code;
}
