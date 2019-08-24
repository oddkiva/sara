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
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>


using namespace std;
using namespace DO::Sara;


void convolution_example()
{
  // Read an image.
  auto image = imread<Rgb32f>(src_path("../../../../data/ksmall.jpg"));

  const auto w = image.width();
  const auto h = image.height();

  // Transpose the image from NHWC to NCHW storage order.
  //                          0123    0312
  auto x = tensor_view(image)
               .reshape(Vector4i{1, h, w, 3})
               .transpose({0, 3, 1, 2});

  // Create the gaussian smoothing kernel for RGB color values.
  auto kt = gaussian_tensor_nchw(4.f, 2);

  // Convolve the image using the GEMM BLAS routine.
  auto y = gemm_convolve(
      x,                  // the signal
      kt,                 // the transposed kernel.
      PeriodicPadding{},  // the padding type
      // make_constant_padding(0.f),      // the padding type
      {1, kt.size(0), 1, 1},  // strides in the convolution
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
}

void convolution_transpose_example()
{
  // Read an image.
  auto image = imread<Rgb32f>(src_path("../../../data/ksmall.jpg"));

  // Upsample the image.
  auto image_resized = upsample(image, 5, 4);

  create_window(image_resized.sizes());
  display(image);
  get_key();
  display(image_resized);
  get_key();
  close_window();
}

GRAPHICS_MAIN()
{
  convolution_example();
  convolution_transpose_example();
  return EXIT_SUCCESS;
}
