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


template <typename T, int N>
Tensor_<T, N> transpose(const TensorView_<T, N>& x, const Matrix<int, N, 1>& order)
{
  Matrix<int, N, 1> out_sizes;
  for (int i = 0; i < N; ++i)
    out_sizes[i] = x.size(order[i]);

  Tensor_<T, N> out{out_sizes};

  auto in_c = x.begin_array();
  Matrix<int, N, 1> out_c = Matrix<int, N, 1>::Zero();

  for ( ; !in_c.end(); ++in_c)
  {
    for (int i = 0; i < N; ++i)
      out_c[i] = in_c.position()[order[i]];

    out(out_c) = *in_c;
  }

  return out;
}

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

int test_grayscale_image()
{
  auto image = Image<float>{};
  imread(image, "/home/david/GitHub/DO-CV/sara/data/sunflowerField.jpg");

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


auto convolve_image_mean33(const Image<Rgb32f>& image)
  -> Image<Rgb32f>
{
  // Get the 3D float row-major tensor view in HxWxC format.
  auto x =
      tensor_view(image).reshape(Vector4i{1, image.height(), image.width(), 3});


  auto convolved_image = Image<Rgb32f>{image.sizes()};
  auto yt = tensor_view(convolved_image)
                .reshape(Vector4i{1, image.height(), image.width(), 3});


  constexpr auto kH = 3;
  constexpr auto kW = 3;
  constexpr auto kC = 3;

  auto phi_x = im2col_strided(x, {1, kH, kW, kC}, {1, 1, 1, kC}, {0, 0, 0, 1});
  cout << "phi = " << phi_x.matrix().rows() << " " << phi_x.matrix().cols()  << endl;

  //                   kH x kW x kI  kO
  Tensor_<float, 2> k{{ 3 *  3 *  3,  3}};

  // Average on the R channel.
  k.matrix().col(0) <<
    1, 0, 0,  1, 0, 0,  1, 0, 0,
    1, 0, 0,  1, 0, 0,  1, 0, 0,
    1, 0, 0,  1, 0, 0,  1, 0, 0;

  // Average on the G channel.
  k.matrix().col(1) <<
    0, 1, 0,  0, 1, 0,  0, 1, 0,
    0, 1, 0,  0, 1, 0,  0, 1, 0,
    0, 1, 0,  0, 1, 0,  0, 1, 0;

  // Average on the B channel.
  k.matrix().col(2) <<
    0, 0, 1,  0, 0, 1,  0, 0, 1,
    0, 0, 1,  0, 0, 1,  0, 0, 1,
    0, 0, 1,  0, 0, 1,  0, 0, 1;
  k.flat_array() /= 9;

  cout << "k = " << k.matrix().rows() << " " << k.matrix().cols()  << endl;
  cout << k.matrix()  << endl;

  auto y = Tensor_<float, 4>{{1, 3, image.height(), image.width()}};
  y.flat_array() = (phi_x.matrix() * k.matrix()).array();

  yt = transpose(y, {0, 2, 3, 1});

  return convolved_image;
}

int test_rgb_image()
{
  // Read an image.
  auto image = Image<Rgb32f>{};
  imread(image, "/home/david/GitHub/DO-CV/sara/data/sunflowerField.jpg");

  auto image_tensor =
      tensor_view(image).reshape(Vector4i{1, image.height(), image.width(), 3});
  // N H W C -> N C H W
  // 0 1 2 3    0 3 1 2
  auto x = transpose(image_tensor, Vector4i{0, 3, 1, 2});

  auto h = image.height();
  auto w = image.width();

  auto kernel = gaussian_kernel(5.f, 2);
  auto kw = kernel.width();
  auto kh = kernel.height();
  auto kc = 3;
  auto ksz = kernel.size();
  auto k = Tensor_<float, 2>{{kh * kw * 3, 3}};
  k.matrix().col(0) << kernel.flat_array(), VectorXf::Zero(ksz), VectorXf::Zero(ksz);
  k.matrix().col(1) << VectorXf::Zero(ksz), kernel.flat_array(), VectorXf::Zero(ksz);
  k.matrix().col(2) << VectorXf::Zero(ksz), VectorXf::Zero(ksz), kernel.flat_array();

  auto phi_x = im2col_strided(x, {1, kc, kh, kw}, {1, kc, 1, 1}, {0, 1, 0, 0});

  auto y = Tensor_<float, 4>{{1, 3, h, w}};
  y.flat_array() = (phi_x.matrix() * k.matrix()).array();

  y = transpose(y, Vector4i{0, 2, 3, 1});

  auto convolved_image = ImageView<Rgb32f>{reinterpret_cast<Rgb32f *>(y.data()), {w, h}};

  create_window(convolved_image.sizes());
  display(image);
  get_key();
  display(convolved_image);
  get_key();
  close_window();

  return EXIT_SUCCESS;
}

GRAPHICS_MAIN()
{
  //return test_grayscale_image();
  return test_rgb_image();
}
