#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN()
{
  create_window(300, 300);
  get_key();

  // We can get the active window with the following function.
  Window w1 = active_window();
  Window w2 = create_window(100, 100);
  set_active_window(w2);
  get_key();
  close_window(w2);

  set_active_window(w1);
  draw_circle(120, 120, 30, Red8);
  get_key();
  close_window();

  return 0;
}