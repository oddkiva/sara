#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{
  Window w = openWindow(300, 300);
  setActiveWindow(w);

  Event e;
  do
  {
    getEvent(1, e);
    fillRect(rand()%300, rand()%300, rand()%50, rand()%50,
      Color3ub(rand()%256, rand()%256, rand()%256));
    //microSleep(100);  // TODO: sometimes if you don't put this, the program
                        // freezes in some machine. Investigate.
  } while (e.key != KEY_ESCAPE);

  cout << "Finished!" << endl;

  closeWindow(getActiveWindow());

  return 0;
}