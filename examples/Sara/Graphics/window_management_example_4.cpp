#include <DO/Sara/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN()
{
  // Open 6 windows aligned in 2x3 grid.
  vector<Window> windows;
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      windows.push_back(
        create_window(200, 200,
                      "Window #" + to_string(i*3+j),
                      300 * j + 300, 300 * i + 50));
      set_active_window(windows.back());
      fill_rect(0, 0,
                get_width(windows.back()),
                get_height(windows.back()),
                Color3ub(rand()%255, rand()%255, rand()%255));
      draw_string(100, 100, to_string(i*3+j), Yellow8, 15);
      cout << "Pressed '" << char(any_get_key()) << "'" << endl;
    }
  }

  // Make the last window active.
  set_active_window(windows.back());
  // Click on any windows to continue.
  any_click();

  // Close the 6 windows in LIFO order.
  for (size_t i = 0; i < windows.size(); ++i)
  {
    any_get_key();
    cout << "Closing window #" << i << endl;
    close_window(windows[i]);
  }

  return 0;
}