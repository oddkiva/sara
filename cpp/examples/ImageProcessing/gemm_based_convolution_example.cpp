// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/MultiArray/InfiniteMultiArrayView.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>


using namespace std;
using namespace DO::Sara;


GRAPHICS_MAIN()
{
  auto image = Image<float>{};
  imread(image, "/home/david/GitHub/DO-CV/sara/data/sunflowerField.jpg");

  // Compute the size of the Gaussian kernel.
  auto gaussian_kernel = [](float sigma, int gauss_truncate)
  {
    auto kernel_size = int(2 * gauss_truncate * sigma + 1);
    // Make sure the Gaussian kernel is at least of size 3 and is of odd size.
    kernel_size = std::max(3, kernel_size);
    if (kernel_size % 2 == 0)
      ++kernel_size;

    // Create the 1D Gaussian kernel.
    auto kernel = Image<float>(kernel_size, kernel_size);
    auto sum = 0.f;

    // Compute the value of the Gaussian and the normalizing factor.
    for (int y = 0; y < kernel_size; ++y)
    {
      const auto v = float(y) - kernel_size / 2.f;
      const auto ky = exp(-v * v / (2.f * sigma * sigma));

      for (int x = 0; x < kernel_size; ++x)
      {
        const auto u = float(x) - kernel_size / 2.f;
        auto kx = exp(-u * u / (2.f * sigma * sigma));
        kernel(x, y) = kx * ky;
        sum += kernel(x, y);
      }
    }

    kernel.flat_array() /= sum;

    return kernel;
  };

  auto kernel = gaussian_kernel(3.f, 2);
  auto convolved_image = Image<float>{image.sizes()};

  auto x = tensor_view(image);
  auto k = tensor_view(kernel);
  auto y = tensor_view(convolved_image);
  gemm_convolve(y, x, k);

  create_window(image.sizes());
  display(kernel.compute<ColorRescale>().compute<Resize>(kernel.sizes() * 20));
  get_key();
  display(convolved_image.compute<ColorRescale>());
  get_key();
  close_window();

  return EXIT_SUCCESS;
}
