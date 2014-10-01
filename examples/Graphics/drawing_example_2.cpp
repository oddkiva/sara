#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{
  // Open a 300x200 window.
  Window W = openWindow(300, 200);
  setAntialiasing(getActiveWindow());
  setTransparency(getActiveWindow());

  drawPoint(Point2f(10.5f, 10.5f), Green8);
  drawPoint(Point2f(20.8f, 52.8132f), Green8);

  drawLine(Point2f(10.5f, 10.5f), Point2f(20.8f, 52.8132f), Blue8, 2);
  drawLine(Point2f(10.5f, 20.5f), Point2f(20.8f, 52.8132f), Magenta8, 5);

  // Draw an oriented ellipse with:
  // center = (150, 100)
  // r1 = 10
  // r2 = 20
  // orientation = 45°
  // in cyan color, and a pencil width = 1.
  drawEllipse(Point2f(150.f, 100.f), 10.f, 20.f, 45.f, Cyan8, 1);
  drawEllipse(Point2f(50.f, 50.f), 10.f, 20.f, 0.f, Red8, 1);

  fillCircle(Point2f(100.f, 100.f), 10.f, Blue8);
  fillEllipse(Point2f(150.f, 150.f), 10.f, 20.f, 72.f, Green8);

  Point2f p1(rand()%300, rand()%200);
  Point2f p2(rand()%300, rand()%200);
  drawPoint((p1*2+p2)/2, Green8);

  click();
  closeWindow(W);

  return 0;
}