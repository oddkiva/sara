#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

int main()
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