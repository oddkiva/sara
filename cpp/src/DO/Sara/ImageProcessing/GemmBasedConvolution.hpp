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

  template <typename T, int N>
  auto im2col(const TensorView_<T, N>& x, const Matrix<int, N, 1>& kernel_sizes,
              const Matrix<int, N, 1>& strides = Matrix<int, N, 1>::Ones(),
              const Matrix<int, N, 1>& shift = Matrix<int, N, 1>::Zero())
      -> Tensor_<T, 2>
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
    const auto num_cols =
        std::accumulate(kernel_sizes.data(), kernel_sizes.data() + N, 1,
                        std::multiplies<int>());

    auto phi_x = Tensor_<T, 2>{num_rows, num_cols};

    for (int r = 0; !xi.end(); ++xi, ++r)
    {
      const Matrix<int, N, 1> s = xi.position() - radius + shift;
      const Matrix<int, N, 1> e =
          xi.position() + radius + Matrix<int, N, 1>::Ones() + shift;

      auto p = patch(x, s, e);

      phi_x.matrix().row(r) = vec(p).transpose();
    }

    return phi_x;
  }

  template <typename T, int N>
  void
  gemm_convolve(TensorView_<T, N>& y,                   //
                const TensorView_<T, N>& x,             //
                const TensorView_<T, N>& k_transposed,  //
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
    auto phi_x = im2col(x, k_sizes, strides, offset);

    y.flat_array() = (phi_x.matrix() * kt.matrix()).array();
  }

  template <typename T, int N>
  auto
  gemm_convolve(const TensorView_<T, N>& x,
                const TensorView_<T, N>& k_transposed,
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
    auto phi_x = im2col(x, k_sizes, strides, offset);

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

} /* namespace Sara */
} /* namespace DO */
