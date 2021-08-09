#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Image/Operations.hpp>
#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  inline auto mean_kernel_2d(int radius) -> Image<float, 2>
  {
    const auto ksize = 2 * radius + 1;

    auto kim = Image<float>{ksize, ksize};
    kim.flat_array().fill(1.f / kim.size());

    return kim;
  }

  inline auto mean_tensor_nchw(int radius)
      -> Tensor_<float, 4>
  {
    const auto kim = mean_kernel_2d(radius);
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

  //! Based on GEMM operation.
  template <typename T, int N, typename Padding>
  auto mean(const TensorView_<T, N>& x,             //
            const Matrix<int, N, 1>& radius,  //
            const Padding& padding,
            const Matrix<int, N, 1>& strides = Matrix<int, N, 1>::Ones(),
            const Matrix<int, N, 1>& shift = Matrix<int, N, 1>::Zero())
      -> Tensor_<T, 2>
  {
    // Create the gaussian smoothing kernel for RGB color values.
    auto kt = mean_tensor_nchw(radius);

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

    return y;
  }

  template <typename T, int N>
  auto ncc(const ImageView<T, N>& f, const ImageView<T, N>& g,
           const ImageView<T, N>& f_mean, const ImageView<T, N>& g_mean,
           const ImageView<T, N>& f_sigma, const ImageView<T, N>& g_sigma)
  {
  }

}
