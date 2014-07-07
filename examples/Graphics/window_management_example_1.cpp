#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

int main()
{
  // Open a 300x200 window.
  Window window = openWindow(300, 200, "A window");

  // A 150x100 filled RED rectangle with top-left corner at (20, 10).
  fillRect(20, 10, 150, 100, Red8);

  // Wait for a click.
  click();

  // Close window.
  closeWindow(window);

  return 0;
}