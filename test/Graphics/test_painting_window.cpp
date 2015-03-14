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
// Qt libraries.
#include <QtTest>
#include <QtWidgets>
// DO-CV libraries.
#include <DO/Graphics/DerivedQObjects/PaintingWindow.hpp>
// Local libraries.
#include "event_scheduler.hpp"

Q_DECLARE_METATYPE(DO::Event)
Q_DECLARE_METATYPE(Qt::Key)
Q_DECLARE_METATYPE(Qt::MouseButtons)

using namespace DO;

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
  PaintingWindow *test_window_;
  QImage true_image_;
  QPainter painter_;

protected: // methods
  TestPaintingWindowDrawingMethods()
  {
    test_window_ = new PaintingWindow(300, 300);
    true_image_ =  QImage(test_window_->size(), QImage::Format_RGB32);
  }

  virtual ~TestPaintingWindowDrawingMethods()
  {
    delete test_window_->scrollArea();
  }

  virtual void SetUp()
  {
    test_window_->setAntialiasing(false);
    test_window_->clear();
    true_image_.fill(Qt::white);
    painter_.begin(&true_image_);
  }

  virtual void TearDown()
  {
    painter_.end();
  }

  QImage get_image_from_window()
  {
    QImage image(test_window_->size(), QImage::Format_RGB32);
    test_window_->render(&image, QPoint(), QRegion(QRect(QPoint(),
      test_window_->size())));
    return image;
  }

};
  
TEST_F(TestPaintingWindowDrawingMethods,
       test_drawPoint_using_integer_coordinates)
{
  int x = 150, y = 100;
  QColor color(255, 0, 0);

  test_window_->drawPoint(x, y, color);

  painter_.setPen(color);
  painter_.drawPoint(QPoint(x, y));

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawPoint_using_QPointF)
{
  double x = 150.95, y = 100.3333;
  QColor color(0, 225, 0);

  test_window_->drawPoint(QPointF(x, y), color);

  painter_.setPen(color);
  painter_.drawPoint(QPointF(x, y));

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawLine_using_integer_coordinates)
{
  int x1 = 100, y1 = 100;
  int x2 = 200, y2 = 240;
  QColor color(0, 255, 0);
  int thickness = 3;

  test_window_->drawLine(x1, y1, x2, y2, color, thickness);

  painter_.setPen(QPen(color, thickness));
  painter_.drawLine(x1, y1, x2, y2);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawLine_using_QPointF)
{
  QPointF p1(100.350, 100.699);
  QPointF p2(203.645, 240.664);
  QColor color(0, 255, 123);
  int thickness = 4;

  test_window_->drawLine(p1, p2, color, thickness);

  painter_.setPen(QPen(color, thickness));
  painter_.drawLine(p1, p2);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawCircle_using_integer_coordinates)
{
  int xc = 150, yc = 123;
  int r = 39;
  QColor color(0, 255, 123);
  int thickness = 4;

  test_window_->drawCircle(xc, yc, r, color, thickness);

  painter_.setPen(QPen(color, thickness));
  painter_.drawEllipse(QPoint(xc, yc), r, r);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawCircle_using_QPointF)
{
  QPointF c(150.999, 123.231);
  int r = 39;
  QColor color(0, 255, 123);
  int thickness = 4;

  test_window_->drawCircle(c, r, color, thickness);

  painter_.setPen(QPen(color, thickness));
  painter_.drawEllipse(c, r, r);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawEllipse_using_axis_aligned_bounding_box)
{
  // Axis-aligned ellipse defined by the following axis-aligned bounding box.
  int x = 150, y = 123;
  int w = 39, h = 100;
  QColor color(0, 255, 123);
  int thickness = 4;

  test_window_->drawEllipse(x, y, w, h, color, thickness);

  painter_.setPen(QPen(color, thickness));
  painter_.drawEllipse(x, y, w, h);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_drawEllipse_with_center_and_semi_axes_and_orientation)
{
  QPointF center(123.123, 156.123);
  qreal r1 = 100.13, r2 = 40.12;
  qreal oriDegree = 48.65;
  QColor color(0, 255, 123);
  int thickness = 4;

  test_window_->drawEllipse(center, r1, r2, oriDegree, color, thickness);

  painter_.setPen(QPen(color, thickness));
  painter_.translate(center);
  painter_.rotate(oriDegree);
  painter_.translate(-r1, -r2);
  painter_.drawEllipse(QRectF(0, 0, 2*r1, 2*r2));

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods, test_drawRect)
{
  int x = 150, y = 123;
  int w = 39, h = 100;
  QColor color(0, 255, 123);
  int thickness = 4;

  test_window_->clear();
  test_window_->drawRect(x, y, w, h, color, thickness);

  painter_.setPen(QPen(color, thickness));
  painter_.drawRect(x, y, w, h);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods, test_drawPoly)
  {
    QPolygonF polygon;
    polygon << QPointF(10, 10) << QPointF(250, 20) << QPointF(150, 258);

    QColor color(0, 255, 123);
    int thickness = 4;

    test_window_->drawPoly(polygon, color, thickness);

    painter_.setPen(QPen(color, thickness));
    painter_.drawPolygon(polygon);

    EXPECT_EQ(get_image_from_window(), true_image_);
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

  test_window_->drawText(x, y, text, color, fontSize, orientation, italic,
    bold, underline);

  painter_.setPen(color);

  QFont font;
  font.setPointSize(fontSize);
  font.setItalic(italic);
  font.setBold(bold);
  font.setUnderline(underline);
  painter_.setFont(font);

  painter_.translate(x, y);
  painter_.rotate(orientation);
  painter_.drawText(0, 0, text);

  EXPECT_EQ(get_image_from_window(), true_image_);
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

  test_window_->drawArrow(x1, y1, x2, y2, color, arrowWidth, arrowHeight,
                          style, width);

  QImage image(get_image_from_window());
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

  test_window_->display(patch, x_offset, y_offset, zoom_factor);

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

  test_window_->fillCircle(xc, yc, r, color);

  QPainterPath path;
  path.addEllipse(qreal(xc) - r/2., qreal(yc) - r/2., r, r);
  painter_.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods, test_fillCircle_using_QPointF)
{
  QPointF c(150.999, 123.231);
  qreal r = 39;
  QColor color(0, 255, 123);

  test_window_->fillCircle(c, r, color);

  QPainterPath path;
  path.addEllipse(c, r, r);
  painter_.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods,
       test_fillEllipse_using_integer_coordinates)
{
  int xc = 150, yc = 123;
  int r1 = 39, r2 = 20;
  QColor color(0, 255, 123);

  test_window_->fillEllipse(xc, yc, r1, r2, color);

  QPainterPath path;
  path.addEllipse(xc, yc, r1, r2);
  painter_.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods, test_fillEllipse_using_QPointF)
{
  QPointF c(150.999, 123.231);
  qreal r1 = 39.01, r2 = 100.29;
  qreal ori_degree = 30.0;
  QColor color(0, 255, 123);

  test_window_->fillEllipse(c, r1, r2, ori_degree, color);

  QPainterPath path;
  path.addEllipse(0, 0, 2*r1, 2*r2);
  painter_.translate(c);
  painter_.rotate(ori_degree);
  painter_.translate(-r1, -r2);
  painter_.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods, test_fillPoly)
{
  QPolygonF polygon;
  polygon << QPointF(10, 10) << QPointF(250, 20) << QPointF(150, 258);

  QColor color(0, 255, 123);

  test_window_->fillPoly(polygon, color);

  QPainterPath path;
  path.addPolygon(polygon);
  painter_.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods, test_fillRect)
{
  int x = 150, y = 123;
  int w = 39, h = 100;
  QColor color(0, 255, 123);

  test_window_->clear();
  test_window_->fillRect(x, y, w, h, color);

  QPainterPath path;
  path.addRect(x, y, w, h);
  painter_.fillPath(path, color);

  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods, test_clear)
{
  test_window_->drawRect(100, 10, 20, 30, QColor(100, 200, 29), 1);

  // "true_image_" is a white image.
  EXPECT_TRUE(get_image_from_window() != true_image_);
  test_window_->clear();
  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods, test_antialiasing)
{
  int x = 150, y = 100, r = 10;
  int penWidth = 3;
  QColor color(255, 0, 0);

  test_window_->setAntialiasing(false);
  test_window_->drawCircle(x, y, r, color, penWidth);

  painter_.setRenderHints(QPainter::Antialiasing, true);
  painter_.setPen(QPen(color, penWidth));
  painter_.drawEllipse(QPointF(x, y), r, r);

  EXPECT_TRUE(get_image_from_window() != true_image_);

  test_window_->clear();
  test_window_->setAntialiasing();
  test_window_->drawCircle(x, y, r, color, penWidth);
  EXPECT_EQ(get_image_from_window(), true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods, test_saveScreen)
{
  QPointF c(150.999, 123.231);
  qreal r1 = 39.01, r2 = 100.29;
  qreal ori_degree = 30.0;
  QColor color(0, 255, 123);

  test_window_->fillEllipse(c, r1, r2, ori_degree, color);
  test_window_->saveScreen("test.png");

  QPainterPath path;
  path.addEllipse(0, 0, 2*r1, 2*r2);
  painter_.translate(c);
  painter_.rotate(ori_degree);
  painter_.translate(-r1, -r2);
  painter_.fillPath(path, color);

  QImage image(QString("test.png"));
  EXPECT_TRUE(!image.isNull());
  EXPECT_EQ(image.size(), true_image_.size());
  EXPECT_EQ(image, true_image_);
}

TEST_F(TestPaintingWindowDrawingMethods, test_transparency)
{
  // TODO: test transparency, which is buggy.
}


class TestPaintingWindowEvents: public testing::Test
{
protected: // data members.
  PaintingWindow *test_window_;
  EventScheduler event_scheduler_;
  QPoint mouse_pos_;
  Qt::Key key_;
  int mouse_buttons_type_id_;
  int event_type_id_;

  int wait_ms_;
  int event_time_ms_;

protected: // methods.
  TestPaintingWindowEvents()
  {
    mouse_buttons_type_id_ = qRegisterMetaType<Qt::MouseButtons>(
      "Qt::MouseButtons"
    );
    event_type_id_ = qRegisterMetaType<Event>("Event");
    test_window_ = new PaintingWindow(300, 300);
    event_scheduler_.set_receiver(test_window_);
    mouse_pos_ = QPoint(10, 10);
    key_ = Qt::Key_A;

    wait_ms_ = 100;
    event_time_ms_ = 10;
  }

  virtual ~TestPaintingWindowEvents()
  {
    delete test_window_->scrollArea();
  }

  void compare_mouse_event(QSignalSpy& spy,
                           const QMouseEvent& expected_event) const
  {
    spy.wait(wait_ms_);
    EXPECT_EQ(spy.count(), 1);

    QList<QVariant> arguments = spy.takeFirst();
    EXPECT_EQ(arguments.at(0).toInt(), expected_event.x());
    EXPECT_EQ(arguments.at(1).toInt(), expected_event.y());
    EXPECT_EQ(arguments.at(2).value<Qt::MouseButtons>(),
              expected_event.buttons());
  }

  void compare_key_event(QSignalSpy& spy) const
  {
    spy.wait(wait_ms_);
    EXPECT_EQ(spy.count(), 1);

    QList<QVariant> arguments = spy.takeFirst();
    EXPECT_EQ(arguments.at(0).toInt(), static_cast<int>(key_));
  }

};

TEST_F(TestPaintingWindowEvents, test_mouse_move_event)
{
  test_window_->setMouseTracking(true);
  QSignalSpy spy(test_window_,
                 SIGNAL(movedMouse(int, int, Qt::MouseButtons)));
  EXPECT_TRUE(spy.isValid());

  QMouseEvent event(
    QEvent::MouseMove, mouse_pos_,
    Qt::NoButton, Qt::NoButton, Qt::NoModifier
  );
  event_scheduler_.schedule_event(&event, event_time_ms_);

  compare_mouse_event(spy, event);
}

TEST_F(TestPaintingWindowEvents, test_mouse_press_event)
{
  QSignalSpy spy(test_window_,
                 SIGNAL(pressedMouseButtons(int, int, Qt::MouseButtons)));
  EXPECT_TRUE(spy.isValid());

  QMouseEvent event(
    QEvent::MouseButtonPress, mouse_pos_,
    Qt::LeftButton, Qt::LeftButton, Qt::NoModifier
  );
  event_scheduler_.schedule_event(&event, event_time_ms_);

  compare_mouse_event(spy, event);
}

TEST_F(TestPaintingWindowEvents, test_mouse_release_event)
{
  QSignalSpy spy(test_window_,
                 SIGNAL(releasedMouseButtons(int, int, Qt::MouseButtons)));
  EXPECT_TRUE(spy.isValid());

  QMouseEvent event(
    QEvent::MouseButtonRelease, mouse_pos_,
    Qt::LeftButton, Qt::LeftButton, Qt::NoModifier
    );
  event_scheduler_.schedule_event(&event, event_time_ms_);

  compare_mouse_event(spy, event);
}

TEST_F(TestPaintingWindowEvents, test_key_press_event)
{
  QSignalSpy spy(test_window_, SIGNAL(pressedKey(int)));
  EXPECT_TRUE(spy.isValid());

  QKeyEvent event(QEvent::KeyPress, key_, Qt::NoModifier);
  event_scheduler_.schedule_event(&event, event_time_ms_);

  compare_key_event(spy);
}

TEST_F(TestPaintingWindowEvents, test_key_release_event)
{
  QSignalSpy spy(test_window_, SIGNAL(releasedKey(int)));
  EXPECT_TRUE(spy.isValid());

  QKeyEvent event(QEvent::KeyRelease, key_, Qt::NoModifier);
  event_scheduler_.schedule_event(&event, event_time_ms_);

  compare_key_event(spy);
}

TEST_F(TestPaintingWindowEvents, test_send_event)
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
