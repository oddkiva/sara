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
#define BOOST_TEST_MODULE "Graphics/GraphicsView Window"

#include <QtTest>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsView.hpp>

#include "event_scheduler.hpp"


Q_DECLARE_METATYPE(DO::Sara::Event)
Q_DECLARE_METATYPE(Qt::Key)
Q_DECLARE_METATYPE(Qt::MouseButtons)


using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestGraphicsView)

BOOST_AUTO_TEST_CASE(test_construction)
{
  const auto width = 300;
  const auto height = 300;
  const auto windowName = QString{"Graphics View Window"};
  const auto x = 200;
  const auto y = 300;

  auto window = new GraphicsView{
    width, height,
    windowName,
    x, y
  };

  BOOST_CHECK_EQUAL(window->width(), width);
  BOOST_CHECK_EQUAL(window->height(), height);
  BOOST_CHECK(window->windowTitle() ==  windowName);
  BOOST_CHECK(window->isVisible());
  BOOST_CHECK(window->lastAddedItem() == nullptr);

  window->deleteLater();
}

BOOST_AUTO_TEST_SUITE_END()

class TestFixtureForGraphicsViewEvents
{
protected:
  GraphicsView *_test_window;
  EventScheduler _event_scheduler;
  QPoint _mouse_pos;
  Qt::Key _key;
  int _mouse_buttons_type_id;
#if QT_VERSION_MAJOR == 6
  QMetaType _event_type;
#else
  int _event_type;
#endif

  int _wait_ms;
  int _event_time_ms;

public:
  TestFixtureForGraphicsViewEvents()
  {
    _mouse_buttons_type_id = qRegisterMetaType<Qt::MouseButtons>(
      "Qt::MouseButtons"
      );
#if QT_VERSION_MAJOR == 6
    _event_type = QMetaType(qRegisterMetaType<Event>("Event"));
#else
    _event_type = qRegisterMetaType<Event>("Event");
#endif
    _test_window = new GraphicsView(300, 300);
    _event_scheduler.set_receiver(_test_window);
    _mouse_pos = QPoint(10, 10);
    _key = Qt::Key_A;

    _wait_ms = 100;
    _event_time_ms = 10;
  }

  virtual ~TestFixtureForGraphicsViewEvents()
  {
    _test_window->deleteLater();
  }

  void compare_key_event(QSignalSpy& spy) const
  {
    spy.wait(500);
    BOOST_CHECK_EQUAL(spy.count(), 1);

    QList<QVariant> arguments = spy.takeFirst();
    BOOST_CHECK_EQUAL(arguments.at(0).toInt(), static_cast<int>(_key));
  }
};

BOOST_FIXTURE_TEST_SUITE(TestGraphicsViewEvents,
                         TestFixtureForGraphicsViewEvents)

BOOST_AUTO_TEST_CASE(test_key_press_event)
{
  QSignalSpy spy{ _test_window, SIGNAL(pressedKey(int)) };
  BOOST_CHECK(spy.isValid());

  auto event = QKeyEvent{ QEvent::KeyPress, _key, Qt::NoModifier };
  _event_scheduler.schedule_event(&event, 10);

  compare_key_event(spy);
}

BOOST_AUTO_TEST_CASE(test_send_no_event)
{
  QSignalSpy spy{ _test_window, SIGNAL(sendEvent(Event)) };
  BOOST_CHECK(spy.isValid());

  QMetaObject::invokeMethod(_test_window, "waitForEvent",
    Qt::AutoConnection, Q_ARG(int, 1));

  // Nothing happens.
  BOOST_CHECK(spy.wait(10));
  BOOST_CHECK_EQUAL(spy.count(), 1);
  auto arguments = spy.takeFirst();
  auto arg = arguments.at(0);
  arg.convert(_event_type);

  const auto event = arguments.at(0).value<Event>();
  BOOST_CHECK(event.type == EventType::NO_EVENT);
}

BOOST_AUTO_TEST_CASE(test_send_pressed_key_event)
{
  // Spy the sendEvent signal.
  QSignalSpy spy{ _test_window, SIGNAL(sendEvent(Event)) };
  BOOST_CHECK(spy.isValid());

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
  BOOST_CHECK(spy.wait(2*wait_ms));

  // Check that the spy received one key press event.
  BOOST_CHECK_EQUAL(spy.count(), 1);

  // Check the details of the key press event.
  auto arguments = spy.takeFirst();
  auto arg = arguments.at(0);
  arg.convert(_event_type);

  const auto event = arguments.at(0).value<Event>();
  BOOST_CHECK(event.type == EventType::KEY_PRESSED);
  BOOST_CHECK_EQUAL(event.key, _key);
}

BOOST_AUTO_TEST_SUITE_END()

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  app.setAttribute(Qt::AA_Use96Dpi, true);

  boost::unit_test::unit_test_main([]() { return true; }, argc, argv);

  return app.exec();
}
