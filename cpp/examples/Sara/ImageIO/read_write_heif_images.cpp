#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>


namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
  const auto image_path = "/home/david/Desktop/Datasets/oddkiva/regents-park/IMG_2708.HEIC";

  const auto image = sara::imread<sara::Rgb8>(image_path);
  sara::create_window(image.sizes());
  sara::display(image);
  sara::get_key();

  sara::imwrite(image, "/home/david/Desktop/test.heic");

  return 0;
}
