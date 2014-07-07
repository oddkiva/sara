#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

int main()
{
  openWindow(300, 300);
  getKey();

  // We can get the active window with the following function.
  Window w1 = getActiveWindow();
  Window w2 = openWindow(100, 100);
  setActiveWindow(w2);
  getKey();
  closeWindow(w2);

  setActiveWindow(w1);
  drawCircle(120, 120, 30, Red8);
  getKey();
  closeWindow();

  return 0;
}