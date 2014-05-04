#include <QtTest>
#include <DO/Graphics/DerivedQObjects/OpenGLWindow.hpp>

using namespace DO;

class TestOpenGLWindow: public QObject
{
  Q_OBJECT
private slots:
  void test_OpenGLWindow_construction()
  {
    int width = 50;
    int height = 50;
    QString windowName = "painting window";
    int x = 200;
    int y = 300;

    OpenGLWindow *window = new OpenGLWindow(width, height,
                                            windowName,
                                            x, y);

  }

};

QTEST_MAIN(TestOpenGLWindow)
#include "test_opengl_window.moc"
