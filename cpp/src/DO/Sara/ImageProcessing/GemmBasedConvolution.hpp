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

  //! @{
  //! @brief Reimplement the `im2row` function.
  template <typename T, int N, typename Padding>
  auto im2row(const TensorView_<T, N>& x,             //
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

      phi_x.matrix().row(r) = p.row_vector();
    }

    return phi_x;
  }
  //! @}

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
    const auto num_cols = std::accumulate(
        sizes.data(), sizes.data() + sizes.size(), 1, std::multiplies<int>());
    const auto num_rows =
        std::accumulate(kernel_sizes.data(), kernel_sizes.data() + N, 1,
                        std::multiplies<int>());

    auto phi_x = Tensor_<T, 2>{num_rows, num_cols};

    for (int c = 0; !xi.end(); ++xi, ++c)
    {
      const Matrix<int, N, 1> s = xi.position() - radius + shift;
      const Matrix<int, N, 1> e =
          xi.position() + radius + Matrix<int, N, 1>::Ones() + shift;

      auto p = Tensor_<T, N>{e - s};
      crop(p, infx, s, e);

      phi_x.matrix().col(c) = p.vector();
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
    const auto phi_x = im2row(x, k_sizes, padding, strides, offset);

    y.colmajor_view()                                                  //
        .reshape(Vector2i{phi_x.matrix().rows(), kt.matrix().cols()})  //
        .matrix() = phi_x.matrix() * kt.matrix();
  }

  template <typename T, int N, typename Padding>
  void
  gemm_convolve_2(TensorView_<T, N>& y,        //
                  const TensorView_<T, N>& x,  //
                  const TensorView_<T, N>& k,  //
                  const Padding& padding,      //
                  const Matrix<int, N, 1>& strides,
                  const Matrix<int, N, 1>& offset = Matrix<int, N, 1>::Zero())
  {
    const auto phi_x = im2col(x, k, padding, strides, offset);

    y.reshape(Vector2i{phi_x.matrix().rows(), k.matrix().cols()}).matrix() =
        k.matrix() * phi_x.matrix();
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
    const auto phi_x = im2row(x, k_sizes, padding, strides, offset);

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

    // We want to enlarge each pixel (u, v) of sizes (1, 1) to a block (u, v) of
    // sizes (kh, kw).
    //
    // If (kh, kw) = (2, 2), then the barycentric coefficients are
    //  k[0][0].matrix() <<
    //   1, 0,
    //   0, 0;
    //  k[0][1].matrix() <<
    //   0.5, 0.5,
    //   0.0, 0.0;
    //  k[1][0].matrix() <<
    //   0.5, 0.0,
    //   0.5, 0.0;
    //  k[1][1].matrix() <<
    //   0.25, 0.25,
    //   0.25, 0.25;
    //
    // It follows that the barycentric weights can be generalized as follows:
    for (int a = 0; a < kh; ++a)
    {
      for (int b = 0; b < kw; ++b)
      {
        auto k_ab = k[a][b].matrix();
        k_ab(0, 0) = float((kh - a) * (kw - b)) / (kh * kw);
        k_ab(1, 1) = float(a * b) / (kh * kw);
        k_ab(0, 1) = float(b * (kh - a)) / (kh * kw);
        k_ab(1, 0) = float((kw - b) * a) / (kh * kw);
      }
    }

    // Reshape the kernel into a 2D matrix.
    auto K = k.reshape(Vector2i{kh * kw, 2 * 2});
    const auto K_matrix = K.matrix();

    // The final tensor is also in CHW format with dilated sizes
    // (kh, kw) o (h, w)  = (kh * h, kw * w).
    auto y = Tensor_<float, 3>{{d, kh * h, kw * w}};

    // Input block from (u, v) to (u + 1, v + 1).
    auto x_block = Tensor_<float, 2>{{2, 2}};
    // Output block from (kw * u, kh * v) -> (kw * (u + 1), kh * (v + 1)).
    auto y_block = Tensor_<float, 2>{{kh, kw}};

    // The input and output blocks viewed as 2D matrices.
    auto x_block_matrix = x_block.matrix();
    auto y_block_matrix = y_block.matrix();
    // The input and output blocks viewed as column vectors.
    auto x_vectorized = x_block.vector();
    auto y_vectorized = y_block.vector();

    // For cache-friendliness, proceed in this order.
    for (int c = 0; c < d; ++c)
    {
      const auto px_slice = px[c].matrix();
      auto y_slice = y[c].matrix();

#pragma omp parallel for
      for (int vu = 0; vu < h * w; ++vu)
      {
        const auto v = vu / w;
        const auto u = vu - v * w;
        // For each channel, grab a (2, 2) input block with top-left corner
        // (u, v).
        x_block_matrix = px_slice.block(v, u, 2, 2);

        // Calculate the (kw, kh) output block with top-left corner
        // (kw * u, kh * v).
        y_vectorized = K_matrix * x_vectorized;

        // Store the result into the output.
        y_slice.block(kh * v, kw * u, kh, kw) = y_block_matrix;
      }
    }

    // Transpose back into interleaved format.
    auto out = Image<Rgb32f>{kw * w, kh * h};
    tensor_view(out) = y.transpose({1, 2, 0});

    return out;
  }

  //! A bit more generalized transposed convolution implementation.
  //! This implementation is simplified (centered, no strides, no offset).
  inline auto transposed_convolution(const Image<Rgb32f>& image,
                                     const TensorView_<float, 4>& k) -> Image<Rgb32f>
  {
    const auto h = image.height();
    const auto w = image.width();
    constexpr auto d = 3;

    // Transpose the image into CHW format.
    auto x = tensor_view(image).transpose({2, 0, 1});
    // Initialize the strided subarray iterator.
    auto infx = make_infinite(x, RepeatPadding{});

    const auto kh_out = k.size(0);
    const auto kw_out = k.size(1);
    const auto kh_in = k.size(2);
    const auto kw_in = k.size(3);

    // Get the extended image data from:
    // - top-left corner: (-kw_out / 2, -kh_out / 2)
    // - bottom-right corner: (w + kw_out / 2, h + kh_out / 2)
    auto px = Tensor_<float, 3>{d, h + kh_out - 1, w + kw_out - 1};
    crop(px, infx,          //
         Vector3i::Zero(),  //
         Vector3i{d, h + kh_out - 1, w + kw_out - 1});

    auto K = k.reshape(Vector2i{kh_out * kw_out, kh_in * kw_in});
    const auto K_matrix = K.matrix();

    // The output tensor in CHW format.
    auto y = Tensor_<float, 3>{{d, kh_out * h, kh_out * w}};

    // Input and output blocks.
    auto x_block = Tensor_<float, 2>{{kh_in, kw_in}};
    auto y_block = Tensor_<float, 2>{{kh_out, kw_out}};

    // The input and output blocks viewed as 2D matrices.
    auto x_block_matrix = x_block.matrix();
    auto y_block_matrix = y_block.matrix();
    // The input and output blocks viewed as column vectors.
    auto x_vectorized = x_block.vector();
    auto y_vectorized = y_block.vector();

    // For cache-friendliness, proceed in this order.
    for (int c = 0; c < d; ++c)
    {
      const auto px_slice = px[c].matrix();
      auto y_slice = y[c].matrix();

#pragma omp parallel for
      for (int vu = 0; vu < h * w; ++vu)
      {
        const auto v = vu / w;
        const auto u = vu - v * w;
        // For each channel, grab a (2, 2) input block with top-left corner
        // (u, v).
        x_block_matrix = px_slice.block(v, u, kh_in, kw_in);

        // Calculate the (kw, kh) output block with top-left corner
        // (kw * u, kh * v).
        y_vectorized = K_matrix * x_vectorized;

        // Store the result into the output.
        y_slice.block(kh_out * v, kw_out * u, kh_out, kw_out) = y_block_matrix;
      }
    }

    // Transpose back to interleaved format.
    auto out = Image<Rgb32f>{kw_out * w, kh_out * h};
    tensor_view(out) = y.transpose({1, 2, 0});

    return out;
  }

} /* namespace Sara */
} /* namespace DO */
