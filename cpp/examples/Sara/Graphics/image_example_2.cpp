// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <random>

#include <DO/Sara/Graphics.hpp>


GRAPHICS_MAIN()
{
  using namespace std;
  using namespace DO::Sara;

  // Uniform distributions.
  auto rng = std::mt19937{};
  auto uni_0_255 = std::uniform_int_distribution<>{0, 255};
  auto uni_0_300 = std::uniform_int_distribution<>{0, 300};

  auto uniform_color_dist = [&]() {
    return Color3ub(uni_0_255(rng), uni_0_255(rng), uni_0_255(rng));
  };

  // Create a white image and display it on a window.
  auto image = Image<Rgb8>{300, 300};
  image.flat_array().fill(White8);

  create_window(300, 300);
  display(image);
  cout << "Display white image" << endl;
  get_key();

  // Draw random colors on an image surface.
  for (auto y = 0; y < image.height(); ++y)
    for (auto x = 0; x < image.width(); ++x)
      draw_point(image, x, y, uniform_color_dist());

  display(image);
  cout << "Display random pixels" << endl;
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
  cout << "Display random rectangles" << endl;
  get_key();
  close_window();

  return 0;
}
