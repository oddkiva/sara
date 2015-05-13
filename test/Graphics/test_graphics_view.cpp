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

// Google Test.
#include <gtest/gtest.h>
// Qt libraries.
#include <QtTest>
// DO-CV libraries.
#include <DO/Sara/Graphics/DerivedQObjects/GraphicsView.hpp>
// Local libraries.
#include "event_scheduler.hpp"

Q_DECLARE_METATYPE(DO::Event)
Q_DECLARE_METATYPE(Qt::Key)
Q_DECLARE_METATYPE(Qt::MouseButtons)

using namespace DO;

TEST(TestGraphicsView, test_construction)
{
  int width = 300;
  int height = 300;
  QString windowName = "Graphics View Window";
  int x = 200;
  int y = 300;

  GraphicsView *window = new GraphicsView(
    width, height,
    windowName,
    x, y
  );

  EXPECT_EQ(window->width(), width);
  EXPECT_EQ(window->height(), height);
  EXPECT_EQ(window->windowTitle(), windowName);
  EXPECT_TRUE(window->isVisible());

  delete window;
}


class TestGraphicsViewEvents: public testing::Test
{
protected: // data members.
  GraphicsView *test_window_;
  EventScheduler event_scheduler_;
  QPoint mouse_pos_;
  Qt::Key key_;
  int mouse_buttons_type_id_;
  int event_type_id_;

protected: // methods.
  TestGraphicsViewEvents()
  {
    mouse_buttons_type_id_ = qRegisterMetaType<Qt::MouseButtons>(
      "Qt::MouseButtons"
      );
    event_type_id_ = qRegisterMetaType<Event>("Event");
    test_window_ = new GraphicsView(300, 300);
    event_scheduler_.set_receiver(test_window_);
    mouse_pos_ = QPoint(10, 10);
    key_ = Qt::Key_A;
  }

  virtual ~TestGraphicsViewEvents()
  {
    delete test_window_;
  }

  void compare_key_event(QSignalSpy& spy) const
  {
    spy.wait(500);
    EXPECT_EQ(spy.count(), 1);

    QList<QVariant> arguments = spy.takeFirst();
    EXPECT_EQ(arguments.at(0).toInt(), static_cast<int>(key_));
  }

};

TEST_F(TestGraphicsViewEvents, test_key_press_event)
{
  QSignalSpy spy(test_window_, SIGNAL(pressedKey(int)));
  EXPECT_TRUE(spy.isValid());

  QKeyEvent event(QEvent::KeyPress, key_, Qt::NoModifier);
  event_scheduler_.schedule_event(&event, 10);

  compare_key_event(spy);
}

TEST_F(TestGraphicsViewEvents, test_send_no_event)
{
  QSignalSpy spy(test_window_, SIGNAL(sendEvent(Event)));
  EXPECT_TRUE(spy.isValid());

  QMetaObject::invokeMethod(test_window_, "waitForEvent",
    Qt::AutoConnection, Q_ARG(int, 1));

  // Nothing happens.
  EXPECT_TRUE(spy.wait(10));
  EXPECT_EQ(spy.count(), 1);
  QList<QVariant> arguments = spy.takeFirst();
  QVariant arg = arguments.at(0);
  arg.convert(event_type_id_);
  Event event(arguments.at(0).value<Event>());
  EXPECT_EQ(event.type, DO::NO_EVENT);
}

TEST_F(TestGraphicsViewEvents, test_send_pressed_key_event)
{
  // Spy the sendEvent signal.
  QSignalSpy spy(test_window_, SIGNAL(sendEvent(Event)));
  EXPECT_TRUE(spy.isValid());

#ifdef _WIN32
  int wait_ms = 100;
  int key_press_time_ms = 10;
#else
  int wait_ms = 10;
  int key_press_time_ms = 1;
#endif

  // Ask the testing window to wait for an event.
  QMetaObject::invokeMethod(test_window_, "waitForEvent",
                            Qt::AutoConnection, Q_ARG(int, wait_ms));

  // Schedule a key press event later.
  QKeyEvent qt_event(QEvent::KeyPress, key_, Qt::NoModifier);
  event_scheduler_.schedule_event(&qt_event, key_press_time_ms);

  // The spy waits for the event.
  EXPECT_TRUE(spy.wait(2*wait_ms));

  // Check that the spy received one key press event.
  EXPECT_EQ(spy.count(), 1);

  // Check the details of the key press event.
  QList<QVariant> arguments = spy.takeFirst();
  QVariant arg = arguments.at(0);
  arg.convert(event_type_id_);
  Event event(arguments.at(0).value<Event>());
  EXPECT_EQ(event.type, DO::KEY_PRESSED);
  EXPECT_EQ(event.key, key_);
}


int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  app.setAttribute(Qt::AA_Use96Dpi, true);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
