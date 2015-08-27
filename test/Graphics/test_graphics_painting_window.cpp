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

#include <QtWidgets>

#include <DO/Sara/Graphics/DerivedQObjects/PaintingWindow.hpp>

#include "event_scheduler.hpp"


Q_DECLARE_METATYPE(DO::Sara::Event)
Q_DECLARE_METATYPE(Qt::Key)
Q_DECLARE_METATYPE(Qt::MouseButtons)

using namespace DO::Sara;


TEST(TestPaintingWindowConstructors,
     test_construction_of_PaintingWindow_with_small_size)
{
  int width = 50;
  int height = 50;
  QString windowName = "painting window";
  int x = 200;
  int y = 300;

  PaintingWindow *window = new PaintingWindow(
    width, height,
    windowName,
    x, y
  );

  EXPECT_EQ(window->width(), width);
  EXPECT_EQ(window->height(), height);
  EXPECT_EQ(window->windowTitle(), windowName);
#ifndef __APPLE__
  // Strangely, when we have 2 monitor screens, this fails on Mac OS X...
  EXPECT_EQ(window->x(), x);
  EXPECT_EQ(window->y(), y);
#endif
  EXPECT_TRUE(window->isVisible());

  delete window->scrollArea();
}

TEST(TestPaintingWindowConstructors,
     test_construction_of_PaintingWindow_with_size_larger_than_desktop)
{
  int width = qApp->desktop()->width();
  int height = qApp->desktop()->height();

  PaintingWindow *window = new PaintingWindow(width, height);

  EXPECT_TRUE(window->scrollArea()->isMaximized());
  EXPECT_TRUE(window->isVisible());

  delete window->scrollArea();
}


class TestPaintingWindowDrawingMethods: public testing::Test
{
protected: // data members
  PaintingWindow *_test_window;
  QImage _true_image;
  QPainter _painter;

protected: // methods
  TestPaintingWindowDrawingMethods()
  {
    _test_window = new PaintingWindow(300, 300);
    _true_image =  QImage(_test_window->size(), QImage::Format_RGB32);
  }

  virtual ~TestPaintingWindowDrawingMethods()
  {
    delete _test_window->scrollArea();
  }

  virtual void SetUp()
  {
    _test_window->setAntialiasing(false);
    _test_window->clear();
    _true_image.fill(Qt::white);
    _painter.begin(&_true_image);
  }

  virtual void TearDown()
  {
    _painter.end();
  }

  QImage get_image_from_window()
  {
    QImage image(_test_window->size(), QImage::Format_RGB32);
    _test_window->render(&image, QPoint(), QRegion(QRect(QPoint(),
                         _test_window->size())));
    return image;
  }

};

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawPoint_using_integer_coordinates)
{
  int x = 150, y = 100;
  QColor color(255, 0, 0);

  _test_window->drawPoint(x, y, color);

  _painter.setPen(color);
  _painter.drawPoint(QPoint(x, y));

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawPoint_using_QPointF)
{
  double x = 150.95, y = 100.3333;
  QColor color(0, 225, 0);

  _test_window->drawPoint(QPointF(x, y), color);

  _painter.setPen(color);
  _painter.drawPoint(QPointF(x, y));

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawLine_using_integer_coordinates)
{
  int x1 = 100, y1 = 100;
  int x2 = 200, y2 = 240;
  QColor color(0, 255, 0);
  int thickness = 3;

  _test_window->drawLine(x1, y1, x2, y2, color, thickness);

  _painter.setPen(QPen(color, thickness));
  _painter.drawLine(x1, y1, x2, y2);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawLine_using_QPointF)
{
  QPointF p1(100.350, 100.699);
  QPointF p2(203.645, 240.664);
  QColor color(0, 255, 123);
  int thickness = 4;

  _test_window->drawLine(p1, p2, color, thickness);

  _painter.setPen(QPen(color, thickness));
  _painter.drawLine(p1, p2);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawCircle_using_integer_coordinates)
{
  int xc = 150, yc = 123;
  int r = 39;
  QColor color(0, 255, 123);
  int thickness = 4;

  _test_window->drawCircle(xc, yc, r, color, thickness);

  _painter.setPen(QPen(color, thickness));
  _painter.drawEllipse(QPoint(xc, yc), r, r);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawCircle_using_QPointF)
{
  QPointF c(150.999, 123.231);
  int r = 39;
  QColor color(0, 255, 123);
  int thickness = 4;

  _test_window->drawCircle(c, r, color, thickness);

  _painter.setPen(QPen(color, thickness));
  _painter.drawEllipse(c, r, r);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawEllipse_using_axis_aligned_bounding_box)
{
  // Axis-aligned ellipse defined by the following axis-aligned bounding box.
  int x = 150, y = 123;
  int w = 39, h = 100;
  QColor color(0, 255, 123);
  int thickness = 4;

  _test_window->drawEllipse(x, y, w, h, color, thickness);

  _painter.setPen(QPen(color, thickness));
  _painter.drawEllipse(x, y, w, h);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawEllipse_with_center_and_semi_axes_and_orientation)
{
  QPointF center(123.123, 156.123);
  qreal r1 = 100.13, r2 = 40.12;
  qreal oriDegree = 48.65;
  QColor color(0, 255, 123);
  int thickness = 4;

  _test_window->drawEllipse(center, r1, r2, oriDegree, color, thickness);

  _painter.setPen(QPen(color, thickness));
  _painter.translate(center);
  _painter.rotate(oriDegree);
  _painter.translate(-r1, -r2);
  _painter.drawEllipse(QRectF(0, 0, 2*r1, 2*r2));

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods, test_drawRect)
{
  int x = 150, y = 123;
  int w = 39, h = 100;
  QColor color(0, 255, 123);
  int thickness = 4;

  _test_window->clear();
  _test_window->drawRect(x, y, w, h, color, thickness);

  _painter.setPen(QPen(color, thickness));
  _painter.drawRect(x, y, w, h);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods, test_drawPoly)
  {
    QPolygonF polygon;
    polygon << QPointF(10, 10) << QPointF(250, 20) << QPointF(150, 258);

    QColor color(0, 255, 123);
    int thickness = 4;

    _test_window->drawPoly(polygon, color, thickness);

    _painter.setPen(QPen(color, thickness));
    _painter.drawPolygon(polygon);

    EXPECT_EQ(get_image_from_window(), _true_image);
  }

TEST_F(TestPaintingWindowDrawingMethods, test_drawText)
{
  // What text.
  QString text("DO-CV is awesome!");
  // Where.
  int x = 130, y = 150;
  double orientation = 36.6;
  // Style.
  QColor color(0, 255, 123);
  int fontSize = 15;
  bool italic = true;
  bool bold = true;
  bool underline = true;

  _test_window->drawText(x, y, text, color, fontSize, orientation, italic,
                         bold, underline);

  _painter.setPen(color);

  QFont font;
  font.setPointSize(fontSize);
  font.setItalic(italic);
  font.setBold(bold);
  font.setUnderline(underline);
  _painter.setFont(font);

  _painter.translate(x, y);
  _painter.rotate(orientation);
  _painter.drawText(0, 0, text);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

// TODO: make this test more exhaustive.
TEST_F(TestPaintingWindowDrawingMethods, test_drawArrow)
{
  int x1 = 150, y1 = 150;
  int x2 = 200, y2 = 200;

  QColor color(0, 255, 123);
  int arrowWidth = 3, arrowHeight = 5;
  int style = 0;
  int width = 3;

  _test_window->drawArrow(x1, y1, x2, y2, color, arrowWidth, arrowHeight,
                          style, width);

  auto image = get_image_from_window();
  for (int x = 150; x < 200; ++x)
    EXPECT_EQ(image.pixel(x,x), color.rgb());
}

TEST_F(TestPaintingWindowDrawingMethods, test_display)
{
  int w = 20, h = 20;
  QImage patch(QSize(w, h), QImage::Format_RGB32);
  patch.fill(Qt::red);
  int x_offset = 20, y_offset = 30;
  double zoom_factor = 2.;

  _test_window->display(patch, x_offset, y_offset, zoom_factor);

  QImage image(get_image_from_window());
  for (int y = 0; y < image.height(); ++y)
    for (int x = 0; x < image.width(); ++x)
      if ( x >= x_offset && x < x_offset+zoom_factor*w &&
           y >= y_offset && y < y_offset+zoom_factor*h )
        EXPECT_EQ(image.pixel(x, y), QColor(Qt::red).rgb());
      else
        EXPECT_EQ(image.pixel(x, y), QColor(Qt::white).rgb());
}

TEST_F(TestPaintingWindowDrawingMethods, test_fillCircle_using_integer_coordinates)
{
  int xc = 150, yc = 123;
  int r = 39;
  QColor color(0, 255, 123);

  _test_window->fillCircle(xc, yc, r, color);

  QPainterPath path;
  path.addEllipse(qreal(xc) - r/2., qreal(yc) - r/2., r, r);
  _painter.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods, test_fillCircle_using_QPointF)
{
  QPointF c(150.999, 123.231);
  qreal r = 39;
  QColor color(0, 255, 123);

  _test_window->fillCircle(c, r, color);

  QPainterPath path;
  path.addEllipse(c, r, r);
  _painter.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_fillEllipse_using_integer_coordinates)
{
  int xc = 150, yc = 123;
  int r1 = 39, r2 = 20;
  QColor color(0, 255, 123);

  _test_window->fillEllipse(xc, yc, r1, r2, color);

  QPainterPath path;
  path.addEllipse(xc, yc, r1, r2);
  _painter.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods, test_fillEllipse_using_QPointF)
{
  QPointF c(150.999, 123.231);
  qreal r1 = 39.01, r2 = 100.29;
  qreal ori_degree = 30.0;
  QColor color(0, 255, 123);

  _test_window->fillEllipse(c, r1, r2, ori_degree, color);

  QPainterPath path;
  path.addEllipse(0, 0, 2*r1, 2*r2);
  _painter.translate(c);
  _painter.rotate(ori_degree);
  _painter.translate(-r1, -r2);
  _painter.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods, test_fillPoly)
{
  QPolygonF polygon;
  polygon << QPointF(10, 10) << QPointF(250, 20) << QPointF(150, 258);

  QColor color(0, 255, 123);

  _test_window->fillPoly(polygon, color);

  QPainterPath path;
  path.addPolygon(polygon);
  _painter.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods, test_fillRect)
{
  int x = 150, y = 123;
  int w = 39, h = 100;
  QColor color(0, 255, 123);

  _test_window->clear();
  _test_window->fillRect(x, y, w, h, color);

  QPainterPath path;
  path.addRect(x, y, w, h);
  _painter.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods, test_clear)
{
  _test_window->drawRect(100, 10, 20, 30, QColor(100, 200, 29), 1);

  // "_true_image" is a white image.
  EXPECT_TRUE(get_image_from_window() != _true_image);
  _test_window->clear();
  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods, test_antialiasing)
{
  int x = 150, y = 100, r = 10;
  int penWidth = 3;
  QColor color(255, 0, 0);

  _test_window->setAntialiasing(false);
  _test_window->drawCircle(x, y, r, color, penWidth);

  _painter.setRenderHints(QPainter::Antialiasing, true);
  _painter.setPen(QPen(color, penWidth));
  _painter.drawEllipse(QPointF(x, y), r, r);

  EXPECT_TRUE(get_image_from_window() != _true_image);

  _test_window->clear();
  _test_window->setAntialiasing();
  _test_window->drawCircle(x, y, r, color, penWidth);
  EXPECT_EQ(get_image_from_window(), _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods, test_saveScreen)
{
  QPointF c(150.999, 123.231);
  qreal r1 = 39.01, r2 = 100.29;
  qreal ori_degree = 30.0;
  QColor color(0, 255, 123);

  _test_window->fillEllipse(c, r1, r2, ori_degree, color);
  _test_window->saveScreen("test.png");

  QPainterPath path;
  path.addEllipse(0, 0, 2*r1, 2*r2);
  _painter.translate(c);
  _painter.rotate(ori_degree);
  _painter.translate(-r1, -r2);
  _painter.fillPath(path, color);

  QImage image(QString("test.png"));
  EXPECT_TRUE(!image.isNull());
  EXPECT_EQ(image.size(), _true_image.size());
  EXPECT_EQ(image, _true_image);
}

TEST_F(TestPaintingWindowDrawingMethods, test_transparency)
{
  // TODO: test transparency, which is buggy.
}


class TestPaintingWindowEvents: public testing::Test
{
protected: // data members.
  PaintingWindow *_test_window;
  EventScheduler _event_scheduler;
  QPoint _mouse_pos;
  Qt::Key _key;
  int _mouse_buttons_type_id;
  int _event_type_id;

  int _wait_ms;
  int _event_time_ms;

protected: // methods.
  TestPaintingWindowEvents()
  {
    _mouse_buttons_type_id = qRegisterMetaType<Qt::MouseButtons>(
      "Qt::MouseButtons"
    );
    _event_type_id = qRegisterMetaType<Event>("Event");
    _test_window = new PaintingWindow(300, 300);
    _event_scheduler.set_receiver(_test_window);
    _mouse_pos = QPoint(10, 10);
    _key = Qt::Key_A;

    _wait_ms = 100;
    _event_time_ms = 10;
  }

  virtual ~TestPaintingWindowEvents()
  {
    delete _test_window->scrollArea();
  }

  void compare_mouse_event(QSignalSpy& spy,
                           const QMouseEvent& expected_event) const
  {
    spy.wait(_wait_ms);
    EXPECT_EQ(spy.count(), 1);

    QList<QVariant> arguments = spy.takeFirst();
    EXPECT_EQ(arguments.at(0).toInt(), expected_event.x());
    EXPECT_EQ(arguments.at(1).toInt(), expected_event.y());
    EXPECT_EQ(arguments.at(2).value<Qt::MouseButtons>(),
              expected_event.buttons());
  }

  void compare_key_event(QSignalSpy& spy) const
  {
    spy.wait(_wait_ms);
    EXPECT_EQ(spy.count(), 1);

    QList<QVariant> arguments = spy.takeFirst();
    EXPECT_EQ(arguments.at(0).toInt(), static_cast<int>(_key));
  }

};

TEST_F(TestPaintingWindowEvents, test_mouse_move_event)
{
  _test_window->setMouseTracking(true);
  QSignalSpy spy(_test_window,
                 SIGNAL(movedMouse(int, int, Qt::MouseButtons)));
  EXPECT_TRUE(spy.isValid());

  QMouseEvent event(
    QEvent::MouseMove, _mouse_pos,
    Qt::NoButton, Qt::NoButton, Qt::NoModifier
  );
  _event_scheduler.schedule_event(&event, _event_time_ms);

  compare_mouse_event(spy, event);
}

TEST_F(TestPaintingWindowEvents, test_mouse_press_event)
{
  QSignalSpy spy(_test_window,
                 SIGNAL(pressedMouseButtons(int, int, Qt::MouseButtons)));
  EXPECT_TRUE(spy.isValid());

  QMouseEvent event(
    QEvent::MouseButtonPress, _mouse_pos,
    Qt::LeftButton, Qt::LeftButton, Qt::NoModifier
  );
  _event_scheduler.schedule_event(&event, _event_time_ms);

  compare_mouse_event(spy, event);
}

TEST_F(TestPaintingWindowEvents, test_mouse_release_event)
{
  QSignalSpy spy(_test_window,
                 SIGNAL(releasedMouseButtons(int, int, Qt::MouseButtons)));
  EXPECT_TRUE(spy.isValid());

  QMouseEvent event(
    QEvent::MouseButtonRelease, _mouse_pos,
    Qt::LeftButton, Qt::LeftButton, Qt::NoModifier
    );
  _event_scheduler.schedule_event(&event, _event_time_ms);

  compare_mouse_event(spy, event);
}

TEST_F(TestPaintingWindowEvents, test_key_press_event)
{
  QSignalSpy spy(_test_window, SIGNAL(pressedKey(int)));
  EXPECT_TRUE(spy.isValid());

  QKeyEvent event(QEvent::KeyPress, _key, Qt::NoModifier);
  _event_scheduler.schedule_event(&event, _event_time_ms);

  compare_key_event(spy);
}

TEST_F(TestPaintingWindowEvents, test_key_release_event)
{
  QSignalSpy spy(_test_window, SIGNAL(releasedKey(int)));
  EXPECT_TRUE(spy.isValid());

  QKeyEvent event(QEvent::KeyRelease, _key, Qt::NoModifier);
  _event_scheduler.schedule_event(&event, _event_time_ms);

  compare_key_event(spy);
}

TEST_F(TestPaintingWindowEvents, test_send_event)
{
  QSignalSpy spy(_test_window, SIGNAL(sendEvent(Event)));
  EXPECT_TRUE(spy.isValid());

  QMetaObject::invokeMethod(_test_window, "waitForEvent",
                            Qt::AutoConnection, Q_ARG(int, 1));

  // Nothing happens.
  EXPECT_TRUE(spy.wait(10));
  EXPECT_EQ(spy.count(), 1);
  QList<QVariant> arguments = spy.takeFirst();
  QVariant arg = arguments.at(0);
  arg.convert(_event_type_id);
  Event event(arguments.at(0).value<Event>());
  EXPECT_EQ(event.type, DO::Sara::NO_EVENT);
}


TEST(TestPaintingWindowResizing,
     test_construction_of_PaintingWindow_with_small_size)
{
  PaintingWindow *window = new PaintingWindow(100, 100);
  EXPECT_EQ(window->width(), 100);
  EXPECT_EQ(window->height(), 100);

  window->resizeScreen(150, 100);
  EXPECT_EQ(window->width(), 150);
  EXPECT_EQ(window->height(), 100);

  delete window->scrollArea();
}

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  app.setAttribute(Qt::AA_Use96Dpi, true);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
