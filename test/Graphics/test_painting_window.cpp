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

#include <QtTest>
#include <QtWidgets>
#include <DO/Graphics/DerivedQObjects/PaintingWindow.hpp>

Q_DECLARE_METATYPE(DO::Event);

using namespace DO;

class TestPaintingWindowConstructors: public QObject
{
  Q_OBJECT

private slots:
  void test_construction_of_PaintingWindow_with_small_size()
  {
    int width = 50;
    int height = 50;
    QString windowName = "painting window";
    int x = 200;
    int y = 300;

    PaintingWindow *window = new PaintingWindow(width, height,
                                                windowName,
                                                x, y);

    QCOMPARE(window->width(), width);
    QCOMPARE(window->height(), height);
    QCOMPARE(window->windowTitle(), windowName);
#ifndef __APPLE__
    // Strangely, when we have 2 monitor screens, this fails on Mac OS X...
    QCOMPARE(window->x(), x);
    QCOMPARE(window->y(), y);
#endif
    QVERIFY(window->isVisible());

    window->deleteLater();
  }

  void test_construction_of_PaintingWindow_with_size_larger_than_desktop()
  {
    int width = qApp->desktop()->width();
    int height = qApp->desktop()->height();

    PaintingWindow *window = new PaintingWindow(width, height);

    QVERIFY(window->scrollArea()->isMaximized());
    QVERIFY(window->isVisible());

    window->deleteLater();
  }
};

class TestPaintingWindowDrawingMethods: public QObject
{
  Q_OBJECT

private: // data members
  PaintingWindow *test_window_;
  QImage true_image_;
  QPainter painter_;

private: // methods
  QImage get_image_from_window()
  {
    QImage image(test_window_->size(), QImage::Format_RGB32);
    test_window_->render(&image, QPoint(), QRegion(QRect(QPoint(),
                         test_window_->size())));
    return image;
  }

private slots:
  void initTestCase()
  {
    test_window_ = new PaintingWindow(300, 300);
    true_image_ =  QImage(test_window_->size(), QImage::Format_RGB32);
  }

  void init()
  {
    test_window_->setAntialiasing(false);
    test_window_->clear();
    true_image_.fill(Qt::white);
    painter_.begin(&true_image_);
  }

  void cleanup()
  {
    painter_.end();
  }

  void cleanupTestCase()
  {
    test_window_->deleteLater();
  }

  void test_drawPoint_using_integer_coordinates()
  {
    int x = 150, y = 100;
    QColor color(255, 0, 0);

    test_window_->drawPoint(x, y, color);

    painter_.setPen(color);
    painter_.drawPoint(QPoint(x, y));

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_drawPoint_using_QPointF()
  {
    double x = 150.95, y = 100.3333;
    QColor color(0, 225, 0);

    test_window_->drawPoint(QPointF(x, y), color);

    painter_.setPen(color);
    painter_.drawPoint(QPointF(x, y));

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_drawLine_using_integer_coordinates()
  {
    int x1 = 100, y1 = 100;
    int x2 = 200, y2 = 240;
    QColor color(0, 255, 0);
    int thickness = 3;

    test_window_->drawLine(x1, y1, x2, y2, color, thickness);

    painter_.setPen(QPen(color, thickness));
    painter_.drawLine(x1, y1, x2, y2);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_drawLine_using_QPointF()
  {
    QPointF p1(100.350, 100.699);
    QPointF p2(203.645, 240.664);
    QColor color(0, 255, 123);
    int thickness = 4;

    test_window_->drawLine(p1, p2, color, thickness);

    painter_.setPen(QPen(color, thickness));
    painter_.drawLine(p1, p2);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_drawCircle_using_integer_coordinates()
  {
    int xc = 150, yc = 123;
    int r = 39;
    QColor color(0, 255, 123);
    int thickness = 4;

    test_window_->drawCircle(xc, yc, r, color, thickness);

    painter_.setPen(QPen(color, thickness));
    painter_.drawEllipse(QPoint(xc, yc), r, r);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_drawCircle_using_QPointF()
  {
    QPointF c(150.999, 123.231);
    int r = 39;
    QColor color(0, 255, 123);
    int thickness = 4;

    test_window_->drawCircle(c, r, color, thickness);

    painter_.setPen(QPen(color, thickness));
    painter_.drawEllipse(c, r, r);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_drawEllipse_using_axis_aligned_bounding_box()
  {
    // Axis-aligned ellipse defined by the following axis-aligned bounding box.
    int x = 150, y = 123;
    int w = 39, h = 100;
    QColor color(0, 255, 123);
    int thickness = 4;

    test_window_->drawEllipse(x, y, w, h, color, thickness);

    painter_.setPen(QPen(color, thickness));
    painter_.drawEllipse(x, y, w, h);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_drawEllipse_with_center_and_semi_axes_and_orientation()
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

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_drawRect()
  {
    int x = 150, y = 123;
    int w = 39, h = 100;
    QColor color(0, 255, 123);
    int thickness = 4;

    test_window_->clear();
    test_window_->drawRect(x, y, w, h, color, thickness);

    painter_.setPen(QPen(color, thickness));
    painter_.drawRect(x, y, w, h);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_drawPoly()
  {
    QPolygonF polygon;
    polygon << QPointF(10, 10) << QPointF(250, 20) << QPointF(150, 258);

    QColor color(0, 255, 123);
    int thickness = 4;

    test_window_->drawPoly(polygon, color, thickness);

    painter_.setPen(QPen(color, thickness));
    painter_.drawPolygon(polygon);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_drawText()
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

    QCOMPARE(get_image_from_window(), true_image_);
  }

  // TODO: make this test more exhaustive.
  void test_drawArrow()
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
      QCOMPARE(image.pixel(x,x), color.rgb());
  }

  void test_display()
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
          QCOMPARE(image.pixel(x, y), QColor(Qt::red).rgb());
        else
          QCOMPARE(image.pixel(x, y), QColor(Qt::white).rgb());
  }

  void test_fillCircle_using_integer_coordinates()
  {
    int xc = 150, yc = 123;
    int r = 39;
    QColor color(0, 255, 123);

    test_window_->fillCircle(xc, yc, r, color);

    QPainterPath path;
    path.addEllipse(qreal(xc) - r/2., qreal(yc) - r/2., r, r);
    painter_.fillPath(path, color);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_fillCircle_using_QPointF()
  {
    QPointF c(150.999, 123.231);
    qreal r = 39;
    QColor color(0, 255, 123);

    test_window_->fillCircle(c, r, color);

    QPainterPath path;
    path.addEllipse(c, r, r);
    painter_.fillPath(path, color);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_fillEllipse_using_integer_coordinates()
  {
    int xc = 150, yc = 123;
    int r1 = 39, r2 = 20;
    QColor color(0, 255, 123);

    test_window_->fillEllipse(xc, yc, r1, r2, color);

    QPainterPath path;
    path.addEllipse(xc, yc, r1, r2);
    painter_.fillPath(path, color);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_fillEllipse_using_QPointF()
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

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_fillPoly()
  {
    QPolygonF polygon;
    polygon << QPointF(10, 10) << QPointF(250, 20) << QPointF(150, 258);

    QColor color(0, 255, 123);

    test_window_->fillPoly(polygon, color);

    QPainterPath path;
    path.addPolygon(polygon);
    painter_.fillPath(path, color);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_fillRect()
  {
    int x = 150, y = 123;
    int w = 39, h = 100;
    QColor color(0, 255, 123);

    test_window_->clear();
    test_window_->fillRect(x, y, w, h, color);

    QPainterPath path;
    path.addRect(x, y, w, h);
    painter_.fillPath(path, color);

    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_clear()
  {
    test_window_->drawRect(100, 10, 20, 30, QColor(100, 200, 29), 1);

    // "true_image_" is a white image.
    QVERIFY(get_image_from_window() != true_image_);
    test_window_->clear();
    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_antialiasing()
  {
    int x = 150, y = 100, r = 10;
    int penWidth = 3;
    QColor color(255, 0, 0);

    test_window_->setAntialiasing(false);
    test_window_->drawCircle(x, y, r, color, penWidth);

    painter_.setRenderHints(QPainter::Antialiasing, true);
    painter_.setPen(QPen(color, penWidth));
    painter_.drawEllipse(QPointF(x, y), r, r);

    QVERIFY(get_image_from_window() != true_image_);

    test_window_->clear();
    test_window_->setAntialiasing();
    test_window_->drawCircle(x, y, r, color, penWidth);
    QCOMPARE(get_image_from_window(), true_image_);
  }

  void test_saveScreen()
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
    QVERIFY(!image.isNull());
    QCOMPARE(image.size(), true_image_.size());    
    QCOMPARE(image, true_image_);
  }

  // TODO: test transparency, which is buggy.
};

class TestPaintingWindowEvents: public QObject
{
  Q_OBJECT

private:
  PaintingWindow *test_window_;
  QPoint mouse_pos_;
  Qt::Key key_;
  int mouse_buttons_type_id_;
  int event_type_id_;

  void compare_spied_mouse_event_arguments(QSignalSpy& spy) const
  {
    QCOMPARE(spy.count(), 1);

    QList<QVariant> arguments = spy.takeFirst();
    QCOMPARE(arguments.at(0).toInt(), mouse_pos_.x());
    QCOMPARE(arguments.at(1).toInt(), mouse_pos_.y());
    QCOMPARE(static_cast<Qt::MouseButtons>(arguments.at(2).toInt()),
             Qt::NoButton);
  }

  void compare_spied_key_event_arguments(QSignalSpy& spy) const
  {
    QCOMPARE(spy.count(), 1);

    QList<QVariant> arguments = spy.takeFirst();
    QCOMPARE(static_cast<Qt::Key>(arguments.at(0).toInt()), key_);
  }

private slots:
  void initTestCase()
  {
    mouse_buttons_type_id_ = qRegisterMetaType<Qt::MouseButtons>(
      "Qt::MouseButtons"
    );
    event_type_id_ = qRegisterMetaType<Event>("Event");
    test_window_ = new PaintingWindow(300, 300);
    mouse_pos_ = QPoint(10, 10);
    key_ = Qt::Key_A;
  }

  void cleanupTestCase()
  {
    test_window_->deleteLater();
  }

  void test_mouse_move_event()
  {
    test_window_->setMouseTracking(true);
    QSignalSpy spy(test_window_,
                   SIGNAL(movedMouse(int, int, Qt::MouseButtons)));
    QVERIFY(spy.isValid());

    QTestEventList events;
    int x = 10, y = 10;
#ifndef _WIN32
    events.addMouseMove(QPoint(x-1, y-1));
#endif
    events.addMouseMove(QPoint(x, y));
    events.simulate(test_window_);

    QTest::mouseMove(test_window_, mouse_pos_);
    compare_spied_mouse_event_arguments(spy);
  }

  void test_mouse_press_event()
  {
    QSignalSpy spy(test_window_,
                   SIGNAL(pressedMouseButtons(int, int, Qt::MouseButtons)));
    QVERIFY(spy.isValid());

    QTest::mousePress(test_window_, Qt::LeftButton, Qt::NoModifier,
                      mouse_pos_);
    compare_spied_mouse_event_arguments(spy);
  }

  void test_mouse_release_event()
  {
    QSignalSpy spy(test_window_,
                   SIGNAL(releasedMouseButtons(int, int, Qt::MouseButtons)));
    QVERIFY(spy.isValid());

    QTest::mouseRelease(test_window_, Qt::LeftButton, Qt::NoModifier,
                        mouse_pos_);
    compare_spied_mouse_event_arguments(spy);
  }

  void test_key_press_event()
  {
    QSignalSpy spy(test_window_, SIGNAL(pressedKey(int)));
    QVERIFY(spy.isValid());

    QTest::keyPress(test_window_, key_, Qt::NoModifier);
    compare_spied_key_event_arguments(spy);
  }

  void test_key_release_event()
  {
    QSignalSpy spy(test_window_, SIGNAL(releasedKey(int)));
    QVERIFY(spy.isValid());

    QTest::keyRelease(test_window_, key_, Qt::NoModifier);
    compare_spied_key_event_arguments(spy);
  }

  void test_send_event()
  {
    QSignalSpy spy(test_window_, SIGNAL(sendEvent(Event)));
    QVERIFY(spy.isValid());

    QMetaObject::invokeMethod(test_window_, "waitForEvent",
                              Qt::AutoConnection, Q_ARG(int, 1));
    
    // Nothing happens.
    QVERIFY(spy.wait(10));
    QCOMPARE(spy.count(), 1);
    QList<QVariant> arguments = spy.takeFirst();
    QVariant arg = arguments.at(0);
    arg.convert(event_type_id_);
    Event event(arguments.at(0).value<Event>());
    QCOMPARE(event.type, DO::NO_EVENT);
  }

};

class TestPaintingWindowResizing: public QObject
{
  Q_OBJECT

private slots:
    void test_construction_of_PaintingWindow_with_small_size()
    {
      PaintingWindow *window = new PaintingWindow(50, 50);
      QCOMPARE(window->width(), 50);
      QCOMPARE(window->height(), 50);

      window->resizeScreen(50, 100);
      QCOMPARE(window->width(), 50);
      QCOMPARE(window->height(), 50);

      window->deleteLater();
    }
};

int main(int argc, char *argv[])
{
  QApplication app(argc, argv);
  app.setAttribute(Qt::AA_Use96Dpi, true);
  QTEST_DISABLE_KEYPAD_NAVIGATION;

  int num_failed_tests = 0;

  TestPaintingWindowConstructors test_painting_window_constructors;
  num_failed_tests += QTest::qExec(&test_painting_window_constructors);

  TestPaintingWindowDrawingMethods test_painting_window_drawing_methods;
  num_failed_tests += QTest::qExec(&test_painting_window_drawing_methods);

#ifndef DISABLE_IN_TRAVIS_CI
  // TODO: this test suite does not seem to work with Travis CI. We probably
  // need to tweak .travis.yml to make it work.
  // For now, this is disabled in travis CI.
  TestPaintingWindowEvents test_painting_window_events;
  num_failed_tests += QTest::qExec(&test_painting_window_events);
#endif

  return num_failed_tests;
}

#include "test_painting_window.moc"
