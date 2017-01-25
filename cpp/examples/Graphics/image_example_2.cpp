#include <DO/Sara/Graphics.hpp>

using namespace std;
using namespace DO::Sara;

GRAPHICS_MAIN()
{
  Image<Rgb8> image(300, 300);
  image.array().fill(White8);

  create_window(300, 300);
  display(image);

  for (int y = 0; y < 300; ++y)
  {
    for (int x = 0; x < 300; ++x)
    {
      Color3ub c(rand()%256, rand()%256, rand()%256);
      draw_point(image, x, y, c);
    }
  }
  display(image);
  get_key();

  image.array().fill(White8);
  display(image);
  get_key();

  for (int i = 0; i <10; ++i)
  {
    int x, y, w, h;
    x = rand()%300;
    y = rand()%300;
    w = rand()%300;
    h = rand()%300;
    Color3ub c(rand()%256, rand()%256, rand()%256);
    fill_rect(image, x, y, w, h, c);
  }
  display(image);
  get_key();
  close_window();

  return 0;
}