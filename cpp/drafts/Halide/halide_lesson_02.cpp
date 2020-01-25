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

  auto image_brighter = image;

  auto input = Halide::Buffer<uint8_t>::make_interleaved(
      reinterpret_cast<uint8_t*>(image.data()), image.width(), image.height(),
      3);

  auto output = Halide::Buffer<uint8_t>::make_interleaved(
      reinterpret_cast<uint8_t*>(image_brighter.data()), image.width(),
      image.height(), 3);

  auto x = Halide::Var{};
  auto y = Halide::Var{};
  auto c = Halide::Var{};

  auto value = input(x, y, c) * 1.5f;
  value = Halide::min(Halide::cast<float>(value), 255.f);
  value = Halide::cast<uint8_t>(value);

  auto brighter = Halide::Func{};
  brighter(x, y, c) = value;
  brighter.output_buffer().dim(0).set_stride(3).dim(2).set_stride(1).set_bounds(
      0, 3);

  // Start the processing.
  auto timer = sara::Timer{};
  timer.restart();
  {
    brighter.realize(output);
    // Copy the result in RGB interleaved format.
  }
  const auto elapsed = timer.elapsed_ms();
  std::cout << "Computation time = " << elapsed << " ms" << std::endl;

  // Show the result.
  sara::display(image_brighter);
  sara::get_key();
  sara::close_window();

  return 0;
}
