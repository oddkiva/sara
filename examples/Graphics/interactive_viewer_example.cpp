#include <DO/Sara/Graphics.hpp>

using namespace std;
using namespace DO::Sara;

GRAPHICS_MAIN()
{
  Image<Rgb8> image;
  load(image, src_path("../../datasets/ksmall.jpg"));

  create_graphics_view(image.width(), image.height());

  for (int i = 0; i < 10; ++i)
  {
    auto pixmap = add_pixmap(image);
    if (pixmap == nullptr)
      cerr << "Error image display" << endl;
  }

  while (get_key() != KEY_ESCAPE);
  close_window();

  return 0;
}