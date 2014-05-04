#include <QtTest>
#include <DO/Graphics/DerivedQObjects/GraphicsView.hpp>

using namespace DO;

class TestGraphicsView: public QObject
{
  Q_OBJECT
private slots:
  void test_construction_of_GraphicsView()
  {
  }
};

QTEST_MAIN(TestGraphicsView)
#include "test_graphics_view.moc"
