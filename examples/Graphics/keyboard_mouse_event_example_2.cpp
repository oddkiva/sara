#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{
  Window w = create_window(300, 300);
  set_active_window(w);

  Event e;
  do
  {
    get_event(1, e);
    fill_rect(rand()%300, rand()%300, rand()%50, rand()%50,
      Color3ub(rand()%256, rand()%256, rand()%256));
    //microSleep(100);  // TODO: sometimes if you don't put this, the program
                        // freezes in some machine. Investigate.
  } while (e.key != KEY_ESCAPE);

  cout << "Finished!" << endl;

  close_window(active_window());

  return 0;
}