#include <QtWidgets>
#include <QtTest>
#include <DO/Graphics/DerivedQObjects/PaintingWindow.hpp>

using namespace DO;

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

  void test_drawPoint()
  {
    PaintingWindow *window = new PaintingWindow(300, 200);
    window->drawPoint(150, 100, QColor(255, 0, 0));

    QImage image(window->size(), QImage::Format_RGB32);
    image.fill(Qt::white);
    window->render(&image, QPoint(), QRegion(QRect(QPoint(), window->size())));
    QCOMPARE(image.pixel(150, 100), QColor(255, 0, 0).rgb());
  }
};

QTEST_MAIN(TestPaintingWindow)
#include "test_painting_window.moc"
