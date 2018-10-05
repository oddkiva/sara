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


template <typename DstArrayView, typename SrcArrayView>
void safe_crop_generic(DstArrayView& dst, const SrcArrayView& src,
                       const typename SrcArrayView::vector_type& begin,
                       const typename SrcArrayView::vector_type& end)
{
  if (dst.sizes() != end - begin)
    throw std::domain_error{"Invalid destination sizes!"};

  auto src_i = src.begin_subarray(begin, end);

  for (auto dst_i = dst.begin(); dst_i != dst.end(); ++src_i, ++dst_i)
    *dst_i = *src_i;
}

template <typename T, int N, typename Padding>
auto im2col_generic(
    const TensorView_<T, N>& x,             //
    const Matrix<int, N, 1>& kernel_sizes,  //
    const Padding& padding,
    const Matrix<int, N, 1>& strides = Matrix<int, N, 1>::Ones(),
    const Matrix<int, N, 1>& shift = Matrix<int, N, 1>::Zero()) -> Tensor_<T, 2>
{
  // Pad sizes must be odd.
  const Matrix<int, N, 1> radius = kernel_sizes / 2;
  const Matrix<int, N, 1> begin = Matrix<int, N, 1>::Zero();
  const Matrix<int, N, 1> end = x.sizes();

  // Initialize the strided subarray iterator.
  auto infx = make_infinite(x, padding);
  auto xi = infx.begin_stepped_subarray(begin, end, strides);

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

    auto p = Tensor_<T, N>{e - s};
    safe_crop_generic(p, infx, s, e);

    phi_x.matrix().row(r) = vec(p).transpose();
  }

  return phi_x;
}


template <typename T, int N, typename Padding>
auto gemm_convolve_generic(
    const TensorView_<T, N>& x,   //
    const TensorView_<T, N>& k_transposed,
    const Padding& padding,
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
  auto phi_x = im2col_generic(x, k_sizes, padding, strides, offset);

  // Determine the sizes of the convolutional output.
  auto y_sizes =
      x.begin_stepped_subarray(Matrix<int, N, 1>::Zero(), x.sizes(), strides)
          .stepped_subarray_sizes();
  y_sizes[1] = kcols;

  // Perform the convolution.
  auto y = Tensor_<T, N>{y_sizes};
  y.colmajor_view()                                                  //
      .reshape(Vector2i{phi_x.matrix().rows(), kt.matrix().cols()})  //
      .matrix() = phi_x.matrix() * kt.matrix();

  return y;
}


// Compute the size of the Gaussian kernel.
auto gaussian_kernel(float sigma, int gauss_truncate) -> Image<float>
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
}

auto gaussian_tensor_nchw(float sigma, int gauss_truncate) -> Tensor_<float, 4>
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

auto upsample_2x2(const Image<Rgb32f>& image)
  -> Image<Rgb32f>
{
  const auto h = image.height();
  const auto w = image.width();
  const auto d = 3;

  // Transpose the image into CHW format.
  auto x = tensor_view(image)
    .reshape(Vector3i{h, w, d})
    .transpose({2, 0, 1});
  // Initialize the strided subarray iterator.
  auto infx = make_infinite(x, PeriodicPadding{});

  // Pad the image.
  auto px = Tensor_<float, 3>{d, h + 1, w + 1};
  safe_crop_generic(px, infx,          //
                    Vector3i::Zero(),  //
                    Vector3i{d, h + 1, w + 1});

  const auto kh = 2;
  const auto kw = 2;
  auto k = Tensor_<float, 4>{{kh, kw, kh, kw}};
  k[0][0].matrix() <<
    1, 0,
    0, 0;
  k[0][1].matrix() <<
    0.5, 0.5,
    0.0, 0.0;
  k[1][0].matrix() <<
    0.5, 0.0,
    0.5, 0.0;
  k[1][1].matrix() <<
    0.25, 0.25,
    0.25, 0.25;

  auto K = Tensor_<float, 4>{{kh, kw * x.size(1), kh, px.size(2)}};
  K.flat_array().fill(0);
  for (int i = 0; i < kh; ++i)
    for (int j = 0; j < kw * x.size(1); ++j)
      K[i][j].matrix().block(0, j / kw, kh, kw) =
          k[i][j % kw].matrix();

  auto K_reshaped = K.reshape(Vector2i{kh * kw * x.size(1), kh * px.size(2)});
  std::cout << "K_reshaped.sizes() = " << K_reshaped.sizes().transpose()
            << std::endl;
  auto px_reshaped = px.reshape(Vector2i{d, px.size(1) * px.size(2)})  //
                         .colmajor_view();
  std::cout << "px_reshaped.sizes() = " << px_reshaped.sizes().transpose()
            << std::endl;

  auto y = Tensor_<float, 3>{{d, kh * x.size(1), kw * x.size(2)}};
  auto y_reshaped = y.reshape(Vector2i{3, kh * x.size(1) * kw * x.size(2)})  //
                        .colmajor_view();
  std::cout << "y_reshaped.sizes() = " << y_reshaped.sizes().transpose()
            << std::endl;

  for (int i = 0; i < x.size(1); ++i)
  {
    y_reshaped.matrix().block(kh * kw * x.size(2) * i,  //
                              0,                        //
                              kh * kw * x.size(2),      //
                              x.size(0)) =
        K_reshaped.matrix() *
        px_reshaped.matrix().block(px.size(2) * i,   //
                                   0,                //
                                   kw * px.size(2),  //
                                   x.size(0));
  }


  auto out = Image<Rgb32f>{kw * w, kh * h};
  tensor_view(out) = y.transpose({1, 2, 0});

  return out;
}


GRAPHICS_MAIN()
{
  // Read an image.
  auto image = Image<Rgb32f>{};
  imread(image, "/home/david/GitHub/DO-CV/sara/data/ksmall.jpg");

  //const auto w = image.width();
  //const auto h = image.height();

  //// Transpose the image from NHWC to NCHW storage order.
  ////                          0123    0312
  //auto x = tensor_view(image)
  //             .reshape(Vector4i{1, h, w, 3})
  //             .transpose({0, 3, 1, 2});

  //// Create the gaussian smoothing kernel for RGB color values.
  //auto kt = gaussian_tensor_nchw(8.f, 2);

  //// Convolve the image using the GEMM BLAS routine.
  //auto y = gemm_convolve_generic(
  //    x,                      // the signal
  //    kt,                     // the transposed kernel.
  //    PeriodicPadding{},      // the padding type
  //    //make_constant_padding(0.f),      // the padding type
  //    {1, kt.size(0), 1, 1},  // strides in the convolution
  //    {0, 1, 0, 0});  // pay attention to the offset here for the C dimension.
  //// Transpose the tensor data back to NHWC storage order to view the image.
  //y = y.transpose({0, 2, 3, 1});

  //auto convolved_image =
  //    ImageView<Rgb32f>{reinterpret_cast<Rgb32f*>(y.data()), {y.size(2), y.size(1)}};

  //create_window(image.sizes());
  //display(image);
  //get_key();
  //display(convolved_image);
  //get_key();
  //close_window();

  auto image_resized = upsample_2x2(image);
  create_window(image_resized.sizes());
  display(image);
  get_key();
  display(image_resized);
  get_key();
  close_window();

  return EXIT_SUCCESS;
}
