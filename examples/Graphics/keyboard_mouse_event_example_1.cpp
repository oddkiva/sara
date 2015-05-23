#include <DO/Sara/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN()
{
  cout << "Basic mouse functions" << endl;

  Window W = create_window(512, 512, "Mouse");
  draw_string(10, 10, "Please click anywhere", Black8);

  click();

  draw_string(10, 40, "click again (left=BLUE, middle=RED, right=done)",
             Black8);

  int button;
  Point2i p;
  while ((button=get_mouse(p)) != MOUSE_RIGHT_BUTTON)
  {
    Rgb8 color;
    if (button == MOUSE_LEFT_BUTTON)
      color = Blue8;
    else if (button == MOUSE_MIDDLE_BUTTON)
      color = Red8;
    else
      color = Black8;
    fill_circle(p, 5, color);
  }

  close_window(W);

  return 0;
}