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

#include <gtest/gtest.h>

#include <QtTest>

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsView.hpp>

#include "event_scheduler.hpp"


Q_DECLARE_METATYPE(DO::Sara::Event)
Q_DECLARE_METATYPE(Qt::Key)
Q_DECLARE_METATYPE(Qt::MouseButtons)


using namespace DO::Sara;


TEST(TestGraphicsView, test_construction)
{
  const auto width = 300;
  const auto height = 300;
  const auto windowName = QString{ "Graphics View Window" };
  const auto x = 200;
  const auto y = 300;

  auto window = new GraphicsView{
    width, height,
    windowName,
    x, y
  };

  EXPECT_EQ(window->width(), width);
  EXPECT_EQ(window->height(), height);
  EXPECT_EQ(window->windowTitle(), windowName);
  EXPECT_TRUE(window->isVisible());
  EXPECT_EQ(window->lastAddedItem(), nullptr);

  delete window;
}


class TestGraphicsViewEvents: public testing::Test
{
protected: // data members.
  GraphicsView *_test_window;
  EventScheduler _event_scheduler;
  QPoint _mouse_pos;
  Qt::Key _key;
  int _mouse_buttons_type_id;
  int _event_type_id;

  int _wait_ms;
  int _event_time_ms;

protected: // methods.
  TestGraphicsViewEvents()
  {
    _mouse_buttons_type_id = qRegisterMetaType<Qt::MouseButtons>(
      "Qt::MouseButtons"
      );
    _event_type_id = qRegisterMetaType<Event>("Event");
    _test_window = new GraphicsView(300, 300);
    _event_scheduler.set_receiver(_test_window);
    _mouse_pos = QPoint(10, 10);
    _key = Qt::Key_A;

    _wait_ms = 100;
    _event_time_ms = 10;
  }

  virtual ~TestGraphicsViewEvents()
  {
    delete _test_window;
  }

  void compare_key_event(QSignalSpy& spy) const
  {
    spy.wait(500);
    EXPECT_EQ(spy.count(), 1);

    QList<QVariant> arguments = spy.takeFirst();
    EXPECT_EQ(arguments.at(0).toInt(), static_cast<int>(_key));
  }

};

TEST_F(TestGraphicsViewEvents, test_key_press_event)
{
  QSignalSpy spy{ _test_window, SIGNAL(pressedKey(int)) };
  EXPECT_TRUE(spy.isValid());

  auto event = QKeyEvent{ QEvent::KeyPress, _key, Qt::NoModifier };
  _event_scheduler.schedule_event(&event, 10);

  compare_key_event(spy);
}

TEST_F(TestGraphicsViewEvents, test_send_no_event)
{
  QSignalSpy spy{ _test_window, SIGNAL(sendEvent(Event)) };
  EXPECT_TRUE(spy.isValid());

  QMetaObject::invokeMethod(_test_window, "waitForEvent",
    Qt::AutoConnection, Q_ARG(int, 1));

  // Nothing happens.
  EXPECT_TRUE(spy.wait(10));
  EXPECT_EQ(spy.count(), 1);
  auto arguments = spy.takeFirst();
  auto arg = arguments.at(0);
  arg.convert(_event_type_id);

  const auto event = arguments.at(0).value<Event>();
  EXPECT_EQ(event.type, DO::Sara::NO_EVENT);
}

TEST_F(TestGraphicsViewEvents, test_send_pressed_key_event)
{
  // Spy the sendEvent signal.
  QSignalSpy spy{ _test_window, SIGNAL(sendEvent(Event)) };
  EXPECT_TRUE(spy.isValid());

#ifdef _WIN32
  auto wait_ms = 100;
  auto key_press_time_ms = 10;
#else
  auto wait_ms = 10;
  auto key_press_time_ms = 1;
#endif

  // Ask the testing window to wait for an event.
  QMetaObject::invokeMethod(_test_window, "waitForEvent",
                            Qt::AutoConnection, Q_ARG(int, wait_ms));

  // Schedule a key press event later.
  QKeyEvent qt_event(QEvent::KeyPress, _key, Qt::NoModifier);
  _event_scheduler.schedule_event(&qt_event, key_press_time_ms);

  // The spy waits for the event.
  EXPECT_TRUE(spy.wait(2*wait_ms));

  // Check that the spy received one key press event.
  EXPECT_EQ(spy.count(), 1);

  // Check the details of the key press event.
  auto arguments = spy.takeFirst();
  auto arg = arguments.at(0);
  arg.convert(_event_type_id);

  const auto event = arguments.at(0).value<Event>();
  EXPECT_EQ(event.type, DO::Sara::KEY_PRESSED);
  EXPECT_EQ(event.key, _key);
}

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  app.setAttribute(Qt::AA_Use96Dpi, true);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
