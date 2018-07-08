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

  //const auto kernel = gaussian_kernel(3.f, 2);
  auto kernel = Image<float>{3, 3};
  kernel.matrix() = Matrix3f::Ones() / 9;

  auto x = tensor_view(image);
  auto k = tensor_view(kernel);

#ifdef DEBUG
  const Vector2i strides{4, 4};
  const auto xi = x.begin_stepped_subarray(Vector2i::Zero(), x.sizes(), strides);
  auto szs = xi.stepped_subarray_sizes();
  std::reverse(szs.data(), szs.data() + szs.size());

  auto convolved_image = Image<float>{szs};
  auto y = tensor_view(convolved_image);

  gemm_convolve_strided(y, x, k, strides);
#else
  // Stride in HxW order.
  const Vector2i strides{2, 2};
  auto y = gemm_convolve_strided(x, k, strides);

  auto convolved_image = image_view(y);
#endif

  create_window(convolved_image.sizes());
  //display(convolved_image.compute<ColorRescale>());
  display(convolved_image);
  get_key();
  close_window();

  return EXIT_SUCCESS;
}
