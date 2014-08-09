#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{
  cout << "Basic mouse functions" << endl;

  Window W = openWindow(512, 512, "Mouse");
  drawString(10, 10, "Please click anywhere", Black8);

  click();  

  drawString(10, 40, "click again (left=BLUE, middle=RED, right=done)",
             Black8);

  int button;    
  Point2i p;
  while ((button=getMouse(p)) != MOUSE_RIGHT_BUTTON)
  {
    Rgb8 color;
    if (button == MOUSE_LEFT_BUTTON)
      color = Blue8;
    else if (button == MOUSE_MIDDLE_BUTTON)
      color = Red8;
    else
      color = Black8;
    fillCircle(p, 5, color);
  }

  closeWindow(W);

  return 0;
}