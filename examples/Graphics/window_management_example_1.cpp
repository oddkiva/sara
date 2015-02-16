#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{
  // Open a 300x200 window.
  Window window = create_window(300, 200, "A window");

  // A 150x100 filled RED rectangle with top-left corner at (20, 10).
  fill_rect(20, 10, 150, 100, Red8);

  // Wait for a click.
  click();

  // Close window.
  close_window(window);

  return 0;
}