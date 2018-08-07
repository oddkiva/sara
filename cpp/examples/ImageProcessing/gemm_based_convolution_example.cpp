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


// Compute the size of the Gaussian kernel.
auto gaussian_kernel(float sigma, int gauss_truncate)
  -> Image<float>
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

auto gaussian_tensor_nchw(float sigma, int gauss_truncate)
  -> Tensor_<float, 4>
{
  const auto kim = gaussian_kernel(sigma, gauss_truncate);
  auto k = kim.flat_array();

  const auto kw = kim.width();
  const auto kh = kim.height();
  const auto ksz = kim.size();
  const auto kin = 3;
  const auto kout = 3;

  auto kt = Tensor_<float, 4>{{kin, kh, kw, kout}};
  auto z = VectorXf::Zero(ksz);

  // Fill in the data.
  auto ktr = kt.reshape(Vector2i{kin * kh * kw, kout});
  // Plane               R  G  B
  ktr.matrix().col(0) << k, z, z;
  ktr.matrix().col(1) << z, k, z;
  ktr.matrix().col(2) << z, z, k;

  return kt;
}

GRAPHICS_MAIN()
{
  // Read an image.
  auto image = Image<Rgb32f>{};
  imread(image, "/home/david/GitHub/DO-CV/sara/data/ksmall.jpg");

  const auto w = image.width();
  const auto h = image.height();

  // Transpose the image from NHWC to NCHW storage order.
  //                          0123    0312
  auto x = tensor_view(image)
               .reshape(Vector4i{1, h, w, 3})
               .transpose({0, 3, 1, 2});

  // Create the gaussian smoothing kernel for RGB color values.
  auto kt = gaussian_tensor_nchw(3.f, 4);

  // Convolve the image using the GEMM BLAS routine.
  auto y = gemm_convolve(
      x,                      // the signal
      kt,                     // the transposed kernel.
      {1, kt.size(0), 2, 2},  // strides in the convolution
      {0, 1, 0, 0});  // pay attention to the offset here for the C dimension.
  // Transpose the tensor data back to NHWC storage order to view the image.
  y = y.transpose({0, 2, 3, 1});

  auto convolved_image =
      ImageView<Rgb32f>{reinterpret_cast<Rgb32f*>(y.data()), {y.size(2), y.size(1)}};

  create_window(image.sizes());
  display(image);
  get_key();
  display(convolved_image);
  get_key();
  close_window();

  return EXIT_SUCCESS;
}
