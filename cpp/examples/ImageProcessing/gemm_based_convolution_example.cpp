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

template <typename T, int N, int O>
void safe_crop2(MultiArrayView<T, N, O>& dst, const MultiArrayView<T, N, O>& src,
               const Matrix<int, N, 1>& begin, const Matrix<int, N, 1>& end,
               const PeriodicPadding& padding)
{
  if (dst.sizes() != end - begin)
    throw std::domain_error{"Invalid destination sizes!"};

  const auto inf_src = make_infinite(src, padding);
  auto src_i = inf_src.begin_subarray(begin, end);

  for (auto dst_i = dst.begin(); dst_i != dst.end(); ++src_i, ++dst_i)
    *dst_i = *src_i;
}

template <typename T, int N, int O>
auto patch2(const MultiArrayView<T, N, O>& in,  //
            const Matrix<int, N, 1>& beg, const Matrix<int, N, 1>& end,
            const PeriodicPadding& pad)
    -> MultiArray<T, N, O>
{
  auto dst = MultiArray<T, N, O>{end - beg};
  safe_crop2(dst, in, begin, end, pad);
  return dst;
}


template <typename T, int N>
auto im2col2(
    const TensorView_<T, N>& x,             //
    const Matrix<int, N, 1>& kernel_sizes,  //
    const PeriodicPadding& padding,
    const Matrix<int, N, 1>& strides = Matrix<int, N, 1>::Ones(),
    const Matrix<int, N, 1>& shift = Matrix<int, N, 1>::Zero()) -> Tensor_<T, 2>
{
  // Pad sizes must be odd.
  const Matrix<int, N, 1> radius = kernel_sizes / 2;
  const Matrix<int, N, 1> begin = Matrix<int, N, 1>::Zero();
  const Matrix<int, N, 1> end = x.sizes();

  // Initialize the strided subarray iterator.
  auto xi = x.begin_stepped_subarray(begin, end, strides);

  const auto sizes = xi.stepped_subarray_sizes();

  // Compute the matrix dimensions.
  const auto num_rows = std::accumulate(
      sizes.data(), sizes.data() + sizes.size(), 1, std::multiplies<int>());
  const auto num_cols = std::accumulate(
      kernel_sizes.data(), kernel_sizes.data() + N, 1, std::multiplies<int>());

  auto phi_x = Tensor_<T, 2>{num_rows, num_cols};

  for (int r = 0; !xi.end(); ++xi, ++r)
  {
    const Matrix<int, N, 1> s = xi.position() - radius + shift;
    const Matrix<int, N, 1> e =
        xi.position() + radius + Matrix<int, N, 1>::Ones() + shift;

    auto p = patch2(x, s, e, padding);

    phi_x.matrix().row(r) = vec(p).transpose();
  }

  return phi_x;
}

template <typename T, int N>
auto gemm_convolve2(const TensorView_<T, N>& x,
                   const TensorView_<T, N>& k_transposed,
                   const PeriodicPadding& padding,
                   const Matrix<int, N, 1>& strides,
                   const Matrix<int, N, 1>& offset = Matrix<int, N, 1>::Zero())
    -> Tensor_<T, N>
{
  const auto& kt_ = k_transposed;
  Matrix<int, N, 1> k_sizes;
  k_sizes << kt_.sizes()[N - 1], kt_.sizes().head(N - 1);

  // Determine the sizes of the kernel.
  const auto krows = std::accumulate(k_sizes.data() + 1, k_sizes.data() + N, 1,
                                     std::multiplies<int>());
  const auto kcols = k_sizes[0];
  auto kt = k_transposed.reshape(Vector2i{krows, kcols});

  // calculate the feature maps for each nd-pixel.
  k_sizes[0] = 1;
  auto phi_x = im2col2(x, k_sizes, padding, strides, offset);

  // Determine the sizes of the convolutional output.
  auto y_sizes =
      x.begin_stepped_subarray(Matrix<int, N, 1>::Zero(), x.sizes(), strides)
          .stepped_subarray_sizes();
  y_sizes[1] = kcols;

  // Perform the convolution.
  auto y = Tensor_<T, N>{y_sizes};
  y.flat_array() = (phi_x.matrix() * kt.matrix()).array();

  return y;
  }
  //! @}

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

  auto pad = PeriodicPadding{};

  // Create the gaussian smoothing kernel for RGB color values.
  auto kt = gaussian_tensor_nchw(3.f, 4);

  // Convolve the image using the GEMM BLAS routine.
  auto y = gemm_convolve2(
      x,                      // the signal
      kt,                     // the transposed kernel.
      pad,                    // the padding type
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

  return EXIT_SUCCESS;
}
