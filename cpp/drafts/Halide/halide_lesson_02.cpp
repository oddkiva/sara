#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>

#include "Halide.h"


namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
  auto image = sara::Image<sara::Rgb8>{};
  if (!sara::load_from_dialog_box(image))
    return EXIT_FAILURE;

  sara::create_window(image.sizes());
  sara::display(image);
  sara::get_key();

  auto image_brighter = image;

  // Start the processing.
  auto timer = sara::Timer{};
  timer.restart();
  {
    auto input = Halide::Buffer<uint8_t>{reinterpret_cast<uint8_t*>(image.data()),
      {image.width(), image.height(), 3}};

    auto brighter = Halide::Func{};

    auto x = Halide::Var{};
    auto y = Halide::Var{};
    auto c = Halide::Var{};

    auto value = input(x, y, c);
    value = Halide::cast<float>(value);
    value *= 1.5f;
    value = Halide::min(value, 255.f);
    value = Halide::cast<uint8_t>(value);

    brighter(x, y, c) = value;

    auto output = Halide::Buffer<uint8_t>{
        reinterpret_cast<uint8_t*>(image_brighter.data()),
        {image.width(), image.height(), 3}};

    brighter.realize(output);
  }
  const auto elapsed = timer.elapsed_ms();
  std::cout << "Computation time = " << elapsed << " ms" << std::endl;

  // Show the result.
  sara::display(image_brighter);
  sara::get_key();

  sara::close_window();

  return 0;
}
