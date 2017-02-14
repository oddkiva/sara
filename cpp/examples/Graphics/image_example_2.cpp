#include <random>

#include <DO/Sara/Graphics.hpp>


GRAPHICS_MAIN()
{
  using namespace std;
  using namespace DO::Sara;

  // Uniform distributions.
  auto rng = std::mt19937{};
  auto uni_0_255 = std::uniform_int_distribution<unsigned char>{};
  auto uni_0_300 = std::uniform_int_distribution<int>{0, 300};

  auto uniform_color_dist = [&]() {
    return Color3ub{uni_0_255(rng), uni_0_255(rng), uni_0_255(rng)};
  };

  // Create a white image and display it on a window.
  auto image = Image<Rgb8>{300, 300};
  image.flat_array().fill(White8);

  create_window(300, 300);
  display(image);
  get_key();

  // Draw random colors on an image surface.
  for (auto y = 0; y < image.height(); ++y)
    for (auto x = 0; x < image.width(); ++x)
      draw_point(image, x, y, uniform_color_dist());

  display(image);
  get_key();


  // Display random rectangles.
  image.flat_array().fill(White8);
  display(image);
  get_key();

  for (int i = 0; i < 10; ++i)
  {
    const auto x = uni_0_300(rng);
    const auto y = uni_0_300(rng);
    const auto w = uni_0_300(rng);
    const auto h = uni_0_300(rng);
    fill_rect(image, x, y, w, h, uniform_color_dist());
  }

  display(image);
  get_key();
  close_window();

  return 0;
}
