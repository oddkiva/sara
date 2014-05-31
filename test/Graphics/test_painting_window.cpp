#include <iostream>
#include <QtTest>
#include <QtWidgets>
#include <DO/Graphics/DerivedQObjects/PaintingWindow.hpp>

using namespace DO;


QImage getImageFromWindow(PaintingWindow *window)
{
  QImage image(window->size(), QImage::Format_RGB32);
  window->render(&image, QPoint(), QRegion(QRect(QPoint(), window->size())));
  return image;
}


class TestPaintingWindow: public QObject
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
  }

  void test_construction_of_PaintingWindow_with_size_larger_than_desktop()
  {
    int width = qApp->desktop()->width();
    int height = qApp->desktop()->height();

    PaintingWindow *window = new PaintingWindow(width, height);

    QVERIFY(window->scrollArea()->isMaximized());
    QVERIFY(window->isVisible());
  }

  void test_drawPoint_using_integer_coordinates()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    int x = 150, y = 100;
    QColor color(255, 0, 0);

    window->drawPoint(x, y, color);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);
    QPainter painter(&trueImage);
    painter.setPen(color);
    painter.drawPoint(QPoint(x, y));

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  void test_drawPoint_using_QPointF()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    double x = 150.95, y = 100.3333;
    QColor color(0, 225, 0);

    window->drawPoint(QPointF(x, y), color);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);
    QPainter painter(&trueImage);
    painter.setPen(color);
    painter.drawPoint(QPointF(x, y));

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  void test_drawLine_using_integer_coordinates()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    int x1 = 100, y1 = 100;
    int x2 = 200, y2 = 240;
    QColor color(0, 255, 0);
    int thickness = 3;

    window->drawLine(x1, y1, x2, y2, color, thickness);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);
    QPainter painter(&trueImage);
    painter.setPen(QPen(color, thickness));
    painter.drawLine(x1, y1, x2, y2);

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  void test_drawLine_using_QPointF()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    QPointF p1(100.350, 100.699);
    QPointF p2(203.645, 240.664);
    QColor color(0, 255, 123);
    int thickness = 4;

    window->drawLine(p1, p2, color, thickness);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);
    QPainter painter(&trueImage);
    painter.setPen(QPen(color, thickness));
    painter.drawLine(p1, p2);

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  void test_drawCircle_using_integer_coordinates()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    int xc = 150, yc = 123;
    int r = 39;
    QColor color(0, 255, 123);
    int thickness = 4;

    window->drawCircle(xc, yc, r, color, thickness);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);
    QPainter painter(&trueImage);
    painter.setPen(QPen(color, thickness));
    painter.drawEllipse(QPoint(xc, yc), r, r);

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  void test_drawCircle_using_Circle_using_QPointF()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    QPointF c(150.999, 123.231);
    int r = 39;
    QColor color(0, 255, 123);
    int thickness = 4;

    window->drawCircle(c, r, color, thickness);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);
    QPainter painter(&trueImage);
    painter.setPen(QPen(color, thickness));
    painter.drawEllipse(c, r, r);

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  void test_drawEllipse_using_axis_aligned_bounding_box()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    // Axis-aligned ellipse defined by the following axis-aligned bounding box.
    int x = 150, y = 123;
    int w = 39, h = 100;
    QColor color(0, 255, 123);
    int thickness = 4;

    window->drawEllipse(x, y, w, h, color, thickness);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);
    QPainter painter(&trueImage);
    painter.setPen(QPen(color, thickness));
    painter.drawEllipse(x, y, w, h);

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  void test_drawEllipse_with_center_and_semi_axes_and_orientation()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    QPointF center(123.123, 156.123);
    qreal r1 = 100.13, r2 = 40.12;
    qreal oriDegree = 48.65;
    QColor color(0, 255, 123);
    int thickness = 4;

    window->drawEllipse(center, r1, r2, oriDegree, color, thickness);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);
    QPainter painter(&trueImage);
    painter.setPen(QPen(color, thickness));
    painter.translate(center);
    painter.rotate(oriDegree);
    painter.translate(-r1, -r2);
    painter.drawEllipse(QRectF(0, 0, 2*r1, 2*r2));

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  void test_drawRect()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    int x = 150, y = 123;
    int w = 39, h = 100;
    QColor color(0, 255, 123);
    int thickness = 4;

    window->drawRect(x, y, w, h, color, thickness);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);
    QPainter painter(&trueImage);
    painter.setPen(QPen(color, thickness));
    painter.drawRect(x, y, w, h);

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  void test_drawPoly()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    QPolygonF polygon;
    polygon << QPointF(10, 10) << QPointF(250, 20) << QPointF(150, 258);

    QColor color(0, 255, 123);
    int thickness = 4;

    window->drawPoly(polygon, color, thickness);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);
    QPainter painter(&trueImage);
    painter.setPen(QPen(color, thickness));
    painter.drawPolygon(polygon);

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  void test_drawText()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

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

    window->drawText(x, y, text, color, fontSize, orientation, italic, bold,
                     underline);

    QImage trueImage(window->size(), QImage::Format_RGB32);
    trueImage.fill(Qt::white);

    QPainter painter(&trueImage);
    painter.setPen(color);

    QFont font;
    font.setPointSize(fontSize);
    font.setItalic(italic);
    font.setBold(bold);
    font.setUnderline(underline);
    painter.setFont(font);

    painter.translate(x, y);
    painter.rotate(orientation);
    painter.drawText(0, 0, text);

    QCOMPARE(getImageFromWindow(window), trueImage);
  }

  // TODO: make this test more exhaustive.
  void test_drawArrow()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    int x1 = 150, y1 = 150;
    int x2 = 200, y2 = 200;

    QColor color(0, 255, 123);
    int arrowWidth = 3, arrowHeight = 5;
    int style = 0;
    int width = 3;

    window->drawArrow(x1, y1, x2, y2, color, arrowWidth, arrowHeight, style,
                      width);

    QImage image(getImageFromWindow(window));
    for (int x = 150; x < 200; ++x)
      QCOMPARE(image.pixel(x,x), color.rgb());
  }

  void test_display()
  {
    PaintingWindow *window = new PaintingWindow(300, 300);

    int w = 20, h = 20;
    QImage patch(QSize(w, h), QImage::Format_RGB32);
    patch.fill(Qt::red);
    int x_offset = 20, y_offset = 30;
    double zoom_factor = 2.;

    window->display(patch, x_offset, y_offset, zoom_factor);

    QImage image(getImageFromWindow(window));
    for (int y = 0; y < image.height(); ++y)
      for (int x = 0; x < image.width(); ++x)
        if ( x >= x_offset && x < x_offset+zoom_factor*w && 
             y >= y_offset && y < y_offset+zoom_factor*h )
          QCOMPARE(image.pixel(x, y), QColor(Qt::red).rgb());
        else
          QCOMPARE(image.pixel(x, y), QColor(Qt::white).rgb());
  }

};


QTEST_MAIN(TestPaintingWindow)
#include "test_painting_window.moc"
