#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>


namespace sara = DO::Sara;


int sara_graphics_main(int argc, char** argv)
{
  const auto image_path =
      argc < 2 ? "/home/david/Desktop/"
                 "snow-leopard-panthera-uncia-closup-1024x768.jpg.webp"
               : argv[2];

  const auto image = sara::imread<sara::Rgb8>(image_path);
  sara::create_window(image.sizes());
  sara::display(image);
  sara::get_key();

  sara::imwrite(image, "/home/david/Desktop/test.webp", 90);

  return 0;
}

int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}
