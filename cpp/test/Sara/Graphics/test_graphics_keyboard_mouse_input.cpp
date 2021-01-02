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

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_MODULE "Graphics/Keyboard and Mouse Input"

#include <boost/test/unit_test.hpp>

#include <vector>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>

#include "event_scheduler.hpp"


using namespace DO::Sara;
using namespace std;


EventScheduler *global_scheduler;


class TestFixtureForKeyboardMouseInputOnSingleWindow
{
protected:
  Window test_window_;

public:
  TestFixtureForKeyboardMouseInputOnSingleWindow()
  {
    test_window_ = create_window(300, 300);
    global_scheduler->set_receiver(test_window_);
  }

  virtual ~TestFixtureForKeyboardMouseInputOnSingleWindow()
  {
    close_window(test_window_);
  }
};


BOOST_FIXTURE_TEST_SUITE(TestKeyboardMouseInputOnSingleWindow,
                         TestFixtureForKeyboardMouseInputOnSingleWindow)

BOOST_AUTO_TEST_CASE(test_get_mouse)
{
  Qt::MouseButton expected_qt_mouse_buttons[] = {
      Qt::LeftButton, Qt::MiddleButton, Qt::RightButton,
  };

  int expected_button_codes[] = {MOUSE_LEFT_BUTTON, MOUSE_MIDDLE_BUTTON,
                                 MOUSE_RIGHT_BUTTON};

  int expected_x = 150, expected_y = 150;

  for (int i = 0; i < 3; ++i)
  {
    Qt::MouseButton input_qt_mouse_button = expected_qt_mouse_buttons[i];
    int expected_button_code = expected_button_codes[i];

    QMouseEvent event(QEvent::MouseButtonRelease,
                      QPointF(expected_x, expected_y), input_qt_mouse_button,
                      Qt::MouseButtons(input_qt_mouse_button), Qt::NoModifier);
    emit get_user_thread().sendEvent(&event, 10);

    int actual_x, actual_y;
    int actual_button = get_mouse(actual_x, actual_y);

    BOOST_CHECK_EQUAL(actual_button, expected_button_code);
    BOOST_CHECK_EQUAL(actual_x, expected_x);
    BOOST_CHECK_EQUAL(actual_y, expected_y);
  }
}

BOOST_AUTO_TEST_CASE(test_click)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent event(QEvent::MouseButtonRelease, QPointF(expected_x, expected_y),
                    expected_button, expected_button, Qt::NoModifier);
  emit get_user_thread().sendEvent(&event, 10);

  click();
}

BOOST_AUTO_TEST_CASE(test_get_key)
{
  int expected_key = Qt::Key_A;
  QKeyEvent event(QEvent::KeyPress, expected_key, Qt::NoModifier);
  emit get_user_thread().sendEvent(&event, 10);

  int actual_key = get_key();

  BOOST_CHECK_EQUAL(actual_key, expected_key);
}

BOOST_AUTO_TEST_CASE(test_get_event_with_no_input_event)
{
  Event event;
  get_event(50, event);
  BOOST_CHECK_EQUAL(event.type, NO_EVENT);
}

BOOST_AUTO_TEST_CASE(test_get_event_with_input_key_event)
{
  int expected_key = Qt::Key_A;
  QKeyEvent qt_event(QEvent::KeyPress, expected_key, Qt::NoModifier);
  emit get_user_thread().sendEvent(&qt_event, 5);

  Event event;
  get_event(50, event);

  BOOST_CHECK_EQUAL(event.type, KEY_PRESSED);
  BOOST_CHECK_EQUAL(event.key, expected_key);
  BOOST_CHECK_EQUAL(event.keyModifiers, static_cast<int>(Qt::NoModifier));
}

BOOST_AUTO_TEST_CASE(test_get_event_with_mouse_pressed_event)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent qt_event(QEvent::MouseButtonPress,
                       QPointF(expected_x, expected_y), expected_button,
                       expected_button, Qt::NoModifier);
  emit get_user_thread().sendEvent(&qt_event, 5);

  Event event;
  get_event(50, event);

  BOOST_CHECK_EQUAL(event.type, MOUSE_PRESSED);
  BOOST_CHECK_EQUAL(event.buttons, 1);
  BOOST_CHECK_EQUAL(event.keyModifiers, static_cast<int>(Qt::NoModifier));
}

BOOST_AUTO_TEST_CASE(test_get_event_with_mouse_released_event)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent qt_event(QEvent::MouseButtonRelease,
                       QPointF(expected_x, expected_y), expected_button,
                       expected_button, Qt::NoModifier);
  emit get_user_thread().sendEvent(&qt_event, 5);

  Event event;
  get_event(50, event);

  BOOST_CHECK_EQUAL(event.type, MOUSE_RELEASED);
  BOOST_CHECK_EQUAL(event.buttons, 1);
  BOOST_CHECK_EQUAL(event.keyModifiers, static_cast<int>(Qt::NoModifier));
}

BOOST_AUTO_TEST_CASE(test_get_event_with_mouse_moved_event)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;
  QMouseEvent qt_event(QEvent::MouseMove, QPointF(expected_x, expected_y),
                       expected_button, expected_button, Qt::NoModifier);
  emit get_user_thread().sendEvent(&qt_event, 5);

  Event event;
  get_event(50, event);

  BOOST_CHECK_EQUAL(event.type, MOUSE_PRESSED_AND_MOVED);
  BOOST_CHECK_EQUAL(event.buttons, 1);
  BOOST_CHECK_EQUAL(event.keyModifiers, static_cast<int>(Qt::NoModifier));
}

BOOST_AUTO_TEST_SUITE_END()


class TestFixtureForKeyboardMouseInputOnAnyWindow
{
protected:
  vector<Window> test_windows_;

  TestFixtureForKeyboardMouseInputOnAnyWindow()
  {
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
        test_windows_.push_back(create_window(200, 200, to_string(3 * i + j),
                                              300 * j + 300, 300 * i + 50));
  }

  virtual ~TestFixtureForKeyboardMouseInputOnAnyWindow()
  {
  }
};

BOOST_FIXTURE_TEST_SUITE(TestKeyboardMouseInputOnAnyWindow,
                         TestFixtureForKeyboardMouseInputOnAnyWindow)

BOOST_AUTO_TEST_CASE(test_any_get_mouse)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;

  for (auto wi : test_windows_)
  {
    set_active_window(wi);

    for (auto wj : test_windows_)
    {
      global_scheduler->set_receiver(wj);
      QMouseEvent event(QEvent::MouseButtonRelease,
                        QPointF(expected_x, expected_y), expected_button,
                        expected_button, Qt::NoModifier);
      emit get_user_thread().sendEvent(&event, 10);

      Point2i actual_position;
      int actual_button = any_get_mouse(actual_position);

      BOOST_CHECK_EQUAL(actual_button, 1);
      BOOST_CHECK_EQUAL(actual_position.x(), expected_x);
      BOOST_CHECK_EQUAL(actual_position.y(), expected_y);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_any_click)
{
  Qt::MouseButton expected_button = Qt::LeftButton;
  int expected_x = 150, expected_y = 150;

  for (auto wi : test_windows_)
  {
    set_active_window(wi);

    for (auto wj : test_windows_)
    {
      global_scheduler->set_receiver(wj);
      QMouseEvent event(QEvent::MouseButtonRelease,
                        QPointF(expected_x, expected_y), expected_button,
                        expected_button, Qt::NoModifier);
      emit get_user_thread().sendEvent(&event, 10);

      any_click();
    }
  }
}

BOOST_AUTO_TEST_CASE(test_any_get_key)
{
  int expected_key = Qt::Key_A;

  for (auto wi : test_windows_)
  {
    set_active_window(wi);

    for (auto wj : test_windows_)
    {
      global_scheduler->set_receiver(wj);
      QKeyEvent event(QEvent::KeyPress, expected_key, Qt::NoModifier);
      emit get_user_thread().sendEvent(&event, 10);

      int actual_key = any_get_key();

      BOOST_CHECK_EQUAL(actual_key, expected_key);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()


int worker_thread(int argc, char** argv)
{
  return boost::unit_test::unit_test_main([]() { return true; }, argc, argv);
}

int main(int argc, char** argv)
{
  // Create Qt Application.
  GraphicsApplication gui_app_(argc, argv);

  // Create an event scheduler on the GUI thread.
  global_scheduler = new EventScheduler;
  // Connect the user thread and the event scheduler.
  QObject::connect(&get_user_thread(), SIGNAL(sendEvent(QEvent*, int)),
                   global_scheduler, SLOT(schedule_event(QEvent*, int)));

  // Run the worker thread.
  gui_app_.register_user_main(worker_thread);
  int return_code = gui_app_.exec();

  // Cleanup and terminate.
  delete global_scheduler;
  return return_code;
}
