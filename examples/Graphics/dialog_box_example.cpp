#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{
  Image<Rgb8> I;
  if (!loadFromDialogBox(I))
    return EXIT_FAILURE;

  openWindow(I.width(), I.height(), "Image loaded from dialog box");
  display(I);
  getKey();

  closeWindow();

  return EXIT_SUCCESS;
}