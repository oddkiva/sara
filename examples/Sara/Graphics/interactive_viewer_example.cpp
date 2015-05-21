#include <DO/Sara/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN()
{
  Image<Rgb8> I;
  load(I, src_path("../../datasets/ksmall.jpg"));

  create_graphics_view(I.width(), I.height());

  for (int i = 0; i < 10; ++i)
  {
    ImageItem image = add_image(I);
    if (!image)
      cerr << "Error image display" << endl;
  }

  while (get_key() != KEY_ESCAPE);
  close_window();

  return 0;
}