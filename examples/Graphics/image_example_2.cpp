#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{
  Image<Rgb8> image(300, 300);
  image.array().fill(White8);

  openWindow(300, 300);
  display(image);

  for (int y = 0; y < 300; ++y)
  {
    for (int x = 0; x < 300; ++x)
    {
      Color3ub c(rand()%256, rand()%256, rand()%256);
      drawPoint(image, x, y, c);
    }
  }
  display(image);
  getKey();

  image.array().fill(White8);
  display(image);
  getKey();

  for (int i = 0; i <10; ++i)
  {
    int x, y, w, h;
    x = rand()%300;
    y = rand()%300;
    w = rand()%300;
    h = rand()%300;
    Color3ub c(rand()%256, rand()%256, rand()%256);
    fillRect(image, x, y, w, h, c);
  }
  display(image);
  getKey();
  closeWindow();

  return 0;
}