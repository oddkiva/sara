#include <QtTest>
#include <QtWidgets>
#include <DO/Graphics/DerivedQObjects/PaintingWindow.hpp>

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
    QCOMPARE(window->x(), x);
    QCOMPARE(window->y(), y);
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

class TestPaintingWindowDrawMethods: public QObject
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

  void test_drawCircle_using_Circle_using_QPointF()
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
};

class TestPaintingWindowEvents: public QObject
{
  Q_OBJECT

private:
  PaintingWindow *test_window_;

private slots:
  void initTestCase()
  {
    qRegisterMetaType<Qt::MouseButtons>("Qt::MouseButtons");
    test_window_ = new PaintingWindow(300, 300);
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
    events.addMouseMove(QPoint(x, y));
    events.addMouseMove(QPoint(x+1, y+1));
    events.simulate(test_window_);

    QCOMPARE(spy.count(), 1);
    QList<QVariant> arguments = spy.takeFirst();

    QCOMPARE(arguments.at(0).toInt(), x+1);
    QCOMPARE(arguments.at(1).toInt(), y+1);
    QCOMPARE(arguments.at(2).toInt(), int(Qt::NoButton));
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

  TestPaintingWindowDrawMethods test_painting_window_draw_methods;
  num_failed_tests += QTest::qExec(&test_painting_window_draw_methods);

  TestPaintingWindowEvents test_painting_window_events;
  num_failed_tests += QTest::qExec(&test_painting_window_events);

  return num_failed_tests;
}

#include "test_painting_window.moc"
