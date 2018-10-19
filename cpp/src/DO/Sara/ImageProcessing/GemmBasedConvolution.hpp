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

//! @file

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Image/Operations.hpp>
#include <DO/Sara/Core/Tensor.hpp>


namespace DO { namespace Sara {

  // In this world everything, everything is **ROW-MAJOR**.
  template <typename T, int N>
  using TensorView_ = TensorView<T, N, RowMajor>;

  template <typename T, int N>
  using Tensor_ = Tensor<T, N, RowMajor>;

  template <typename T, int N>
  auto image_view(TensorView_<T, N> in) -> ImageView<T, N>
  {
    auto out_sizes = in.sizes();
    std::reverse(out_sizes.data(), out_sizes.data() + N);
    return ImageView<T, N>{in.data(), out_sizes};
  }

  template <typename T, int N>
  auto patch(const TensorView_<T, N>& in, const Matrix<int, N, 1>& beg,
             const Matrix<int, N, 1>& end) -> Image<T, N>
  {
    auto bb = beg;
    auto ee = end;
    std::reverse(bb.data(), bb.data() + N);
    std::reverse(ee.data(), ee.data() + N);
    return safe_crop(image_view(in), bb, ee);
  }

  template <typename T, int N, int StorageOrder>
  auto vec(MultiArrayView<T, N, StorageOrder>& in)
      -> Map<Matrix<typename ElementTraits<T>::value_type, Dynamic, 1>>
  {
    return {reinterpret_cast<typename ElementTraits<T>::pointer>(in.data()),
            static_cast<int64_t>(in.size())};
  }

  template <typename T, int N, int StorageOrder>
  auto vec(const MultiArrayView<T, N, StorageOrder>& in)
      -> Map<const Matrix<typename ElementTraits<T>::value_type, Dynamic, 1>>
  {
    return {reinterpret_cast<typename ElementTraits<T>::const_pointer>(in.data()),
            static_cast<int64_t>(in.size())};
  }

  //! @{
  //! @brief Reimplement the `im2col` function.
  template <typename T, int N, typename Padding>
  auto im2col(const TensorView_<T, N>& x,             //
              const Matrix<int, N, 1>& kernel_sizes,  //
              const Padding& padding,
              const Matrix<int, N, 1>& strides = Matrix<int, N, 1>::Ones(),
              const Matrix<int, N, 1>& shift = Matrix<int, N, 1>::Zero())
      -> Tensor_<T, 2>
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
    const auto num_cols =
        std::accumulate(kernel_sizes.data(), kernel_sizes.data() + N, 1,
                        std::multiplies<int>());

    auto phi_x = Tensor_<T, 2>{num_rows, num_cols};

    for (int r = 0; !xi.end(); ++xi, ++r)
    {
      const Matrix<int, N, 1> s = xi.position() - radius + shift;
      const Matrix<int, N, 1> e =
          xi.position() + radius + Matrix<int, N, 1>::Ones() + shift;

      auto p = Tensor_<T, N>{e - s};
      crop(p, infx, s, e);

      phi_x.matrix().row(r) = vec(p).transpose();
    }

    return phi_x;
  }
  //! @}


  //! @{
  //! @brief Apply the GEMM-based convolution.
  /*! The implementation is valid only for interleaved data:
   *  - NCHW,   image stored as interleaved channel.
   *  - NCDHW,  volumic data stored as interleaved channel.
   *  - NCTHW,  2d video interleaved planar data.
   *  - NCTDHW  3d video interleaved planar data.
   */
  template <typename T, int N, typename Padding>
  void
  gemm_convolve(TensorView_<T, N>& y,                   //
                const TensorView_<T, N>& x,             //
                const TensorView_<T, N>& k_transposed,  //
                const Padding& padding,                 //
                const Matrix<int, N, 1>& strides,
                const Matrix<int, N, 1>& offset = Matrix<int, N, 1>::Zero())
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
    auto phi_x = im2col(x, k_sizes, padding, strides, offset);

    y.colmajor_view()                                                  //
        .reshape(Vector2i{phi_x.matrix().rows(), kt.matrix().cols()})  //
        .matrix() = phi_x.matrix() * kt.matrix();
  }

  template <typename T, int N, typename Padding>
  auto
  gemm_convolve(const TensorView_<T, N>& x,             //
                const TensorView_<T, N>& k_transposed,  //
                const Padding& padding,                 //
                const Matrix<int, N, 1>& strides,
                const Matrix<int, N, 1>& offset = Matrix<int, N, 1>::Zero())
      -> Tensor_<T, N>
  {
    const auto& kt_ = k_transposed;
    Matrix<int, N, 1> k_sizes;
    k_sizes << kt_.sizes()[N - 1], kt_.sizes().head(N - 1);

    // Determine the sizes of the kernel.
    const auto krows = std::accumulate(k_sizes.data() + 1, k_sizes.data() + N,
                                       1, std::multiplies<int>());
    const auto kcols = k_sizes[0];
    auto kt = k_transposed.reshape(Vector2i{krows, kcols});

    // calculate the feature maps for each nd-pixel.
    k_sizes[0] = 1;
    auto phi_x = im2col(x, k_sizes, padding, strides, offset);

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
  //! @}

} /* namespace Sara */
} /* namespace DO */


// Useful examples and applications.
namespace DO { namespace Sara {

  //!@ {
  //! Compute the size of the Gaussian kernel.
  inline auto gaussian_kernel(float sigma, int gauss_truncate) -> Image<float>
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

  inline auto gaussian_tensor_nchw(float sigma, int gauss_truncate)
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


  //! Particular application of the transposed convolution.
  inline auto upsample(const Image<Rgb32f>& image, int kh, int kw)
      -> Image<Rgb32f>
  {
    const auto h = image.height();
    const auto w = image.width();
    constexpr auto d = 3;

    // Transpose the image into CHW format.
    auto x = tensor_view(image).transpose({2, 0, 1});
    // Initialize the strided subarray iterator.
    auto infx = make_infinite(x, RepeatPadding{});

    // Pad the image.
    auto px = Tensor_<float, 3>{d, h + kh - 1, w + kw - 1};
    crop(px, infx,          //
         Vector3i::Zero(),  //
         Vector3i{d, h + kh - 1, w + kw - 1});

    // List the interpolation filters.
    auto k = Tensor_<float, 4>{{kh, kw, 2, 2}};
    k.flat_array().fill(0);
    // k[0][0].matrix() <<
    //  1, 0,
    //  0, 0;
    // k[0][1].matrix() <<
    //  0.5, 0.5,
    //  0.0, 0.0;
    // k[1][0].matrix() <<
    //  0.5, 0.0,
    //  0.5, 0.0;
    // k[1][1].matrix() <<
    //  0.25, 0.25,
    //  0.25, 0.25;

    for (int a = 0; a < kh; ++a)
      for (int b = 0; b < kw; ++b)
      {
        k[a][b].matrix()(0, 0) = float((kh - a) * (kw - b)) / (kh * kw);
        k[a][b].matrix()(1, 1) = float(a * b) / (kh * kw);
        k[a][b].matrix()(0, 1) = float(b * (kh - a)) / (kh * kw);
        k[a][b].matrix()(1, 0) = float((kw - b) * a) / (kh * kw);
      }


    auto K = k.reshape(Vector2i{kh * kw, 2 * 2});
    std::cout << K.matrix() << std::endl;

    auto y = Tensor_<float, 3>{{d, kh * h, kw * w}};

    auto y_block = Tensor_<float, 3>{{d, kh, kw}};
    auto x_block = Tensor_<float, 3>{{d, 2, 2}};

    auto y_block_2 = y_block.reshape(Vector2i{d, kh * kw}).colmajor_view();
    auto x_block_2 = x_block.reshape(Vector2i{d, 2 * 2}).colmajor_view();

    for (int v = 0; v < h; ++v)
      for (int u = 0; u < w; ++u)
      {
        for (int c = 0; c < d; ++c)
          x_block[c].matrix() = px[c].matrix().block(v, u, kh, kw);

        y_block_2.matrix() = K.matrix() * x_block_2.matrix();

        for (int c = 0; c < d; ++c)
          y[c].matrix().block(kh * v, kw * u, kh, kw) = y_block[c].matrix();
      }

    auto out = Image<Rgb32f>{kw * w, kh * h};
    tensor_view(out) = y.transpose({1, 2, 0});

    return out;
  }

  inline auto transposed_convolution(const Image<Rgb32f>& image, int kh, int kw)
      -> Image<Rgb32f>
  {
    const auto h = image.height();
    const auto w = image.width();
    constexpr auto d = 3;

    // Transpose the image into CHW format.
    auto x = tensor_view(image).transpose({2, 0, 1});
    // Initialize the strided subarray iterator.
    auto infx = make_infinite(x, RepeatPadding{});

    // Pad the image.
    auto px = Tensor_<float, 3>{d, h + kh - 1, w + kw - 1};
    crop(px, infx,          //
         Vector3i::Zero(),  //
         Vector3i{d, h + kh - 1, w + kw - 1});

    // List the interpolation filters.
    auto k = Tensor_<float, 4>{{kh, kw, 2, 2}};
    k.flat_array().fill(0);
    for (int a = 0; a < kh; ++a)
      for (int b = 0; b < kw; ++b)
      {
        k[a][b].matrix()(0, 0) = float((kh - a) * (kw - b)) / (kh * kw);
        k[a][b].matrix()(1, 1) = float(a * b) / (kh * kw);
        k[a][b].matrix()(0, 1) = float(b * (kh - a)) / (kh * kw);
        k[a][b].matrix()(1, 0) = float((kw - b) * a) / (kh * kw);
      }

    auto K = k.reshape(Vector2i{kh * kw, 2 * 2});

    auto y = Tensor_<float, 3>{{d, kh * h, kw * w}};

    auto y_block = Tensor_<float, 3>{{d, kh, kw}};
    auto x_block = Tensor_<float, 3>{{d, 2, 2}};

    auto y_block_2 = y_block.reshape(Vector2i{d, kh * kw}).colmajor_view();
    auto x_block_2 = x_block.reshape(Vector2i{d, 2 * 2}).colmajor_view();

    for (int v = 0; v < h; ++v)
      for (int u = 0; u < w; ++u)
      {
        for (int c = 0; c < d; ++c)
          x_block[c].matrix() = px[c].matrix().block(v, u, kh, kw);

        y_block_2.matrix() = K.matrix() * x_block_2.matrix();

        for (int c = 0; c < d; ++c)
          y[c].matrix().block(kh * v, kw * u, kh, kw) = y_block[c].matrix();
      }

    auto out = Image<Rgb32f>{kw * w, kh * h};
    tensor_view(out) = y.transpose({1, 2, 0});

    return out;
  }
} /* namespace Sara */
} /* namespace DO */
