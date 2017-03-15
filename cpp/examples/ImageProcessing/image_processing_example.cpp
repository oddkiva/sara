#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>

using namespace std;
using namespace DO::Sara;


GRAPHICS_MAIN()
{
  Image<Rgb8> image;
  if (!load_from_dialog_box(image))
    return EXIT_FAILURE;

  // Resize image.
  create_window((image.sizes() * 2).eval(), "Image loaded from dialog box");
  display(enlarge(image, 2));
  display(image);
  display(reduce(image, 2));
  get_key();

  // Pixelwise operations.
  auto res = image.cwise_transform([](const Rgb8& color) {
    Rgb64f color_64f;
    smart_convert_color(color, color_64f);
    color_64f =
    color_64f.cwiseProduct(color_64f);
    return color_64f;
  });

  display(res);
  get_key();

  close_window();

  return EXIT_SUCCESS;
}
