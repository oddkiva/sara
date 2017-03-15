#include <DO/Sara/Graphics.hpp>

using namespace std;
using namespace DO::Sara;

GRAPHICS_MAIN()
{
  Image<Rgb8> image;
  if (!load_from_dialog_box(image))
    return EXIT_FAILURE;

  create_window(image.width(), image.height(), "Image loaded from dialog box");
  display(image);
  get_key();

  close_window();

  return EXIT_SUCCESS;
}