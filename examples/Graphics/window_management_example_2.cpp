#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{ 
  // Open a 300x200 window with top-left corner (10,10).
  Window w1 = openWindow(300, 200, "A first window", 10, 10);
  
  // Draw a 150x100 red rectangle with top-left corner at (20, 10).
  drawRect(20, 10, 150, 100, Red8);

  // Create a second window with dimensions 200x300 with top-left corner (320, 10)
  Window w2 = openWindow(200, 300, "A second window", 330, 10);
  // To draw on the second window, we need to tell the computer that we want
  // "activate" it.
  setActiveWindow(w2);
  // Draw a blue line from coordinates (20, 10) to coordinates (150, 270).
  drawLine(20, 10, 150, 270, Blue8);
  
  // Draw another green line but on the first window.
  setActiveWindow(w1);
  drawLine(20, 10, 250, 170, Green8);
  
  // Wait for a click in any window.
  anyClick();
  
  // It is OK if we forget to close the windows, there will be no memory leak.

  return 0;
}