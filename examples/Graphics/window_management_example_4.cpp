#include <DO/Core/Stringify.hpp>
#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

int main()
{
  // Open 6 windows aligned in 2x3 grid.
  vector<Window> windows;
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      windows.push_back(
        openWindow(200, 200,
                   "Window #" + toString(i*3+j),
                   300*j+300, 300*i+50));
      setActiveWindow(windows.back());
      fillRect(0, 0,
               getWindowWidth(windows.back()),
               getWindowHeight(windows.back()),
               Color3ub(rand()%255, rand()%255, rand()%255));
      drawString(100, 100, toString(i*3+j), Yellow8, 15);
      cout << "Pressed '" << char(anyGetKey()) << "'" << endl;
    }
  }

  // Make the last window active.
  setActiveWindow(windows.back());
  // Click on any windows to continue.
  anyClick(); 

  // Close the 6 windows in LIFO order.
  for (size_t i = 0; i < windows.size(); ++i)
  {
    anyGetKey();
    cout << "Closing window #" << i << endl;
    closeWindow(windows[i]);
  }

  return 0;
}