#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

int main()
{
  cout << "Advanced event handling" << endl;
  Window W = openWindow(1024, 768);
  setActiveWindow(W);
  Image<Color3ub> I;
  load(I, srcPath("../../datasets/ksmall.jpg"));

  Event ev;
  do {
    getEvent(500,ev); // Wait an event (return if no event for 500ms)
    switch (ev.type){
    case NO_EVENT:
      break;
    case MOUSE_PRESSED_AND_MOVED:
      clearWindow();
      display(I, ev.mousePos-I.sizes()*3/8, 0.75);
      cout << "Mouse moved. Position = " << endl << ev.mousePos << endl;
      break;
    case KEY_PRESSED:
      cout << "Key " << ev.key << " pressed"<< endl;
      break;
    case KEY_RELEASED:
      cout << "Key " << ev.key << " released"<< endl;
      break;
    case MOUSE_PRESSED:
      clearWindow();
      display(I, ev.mousePos-I.sizes()*3/8, 0.75);
      cout << "Button " << ev.buttons << " pressed"<< endl;
      break;
    case MOUSE_RELEASED:
      cout << "Button " << ev.buttons << " released"<< endl;
      break;
    }
  } while (ev.type != KEY_PRESSED || ev.key != KEY_UP);
  closeWindow(W);

  return 0;
}