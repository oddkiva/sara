#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Core/Image/Operations.hpp>


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
    return safe_crop(image_view(in), beg, end);
  }

  template <typename T, int N>
  auto vec(const TensorView_<T, N>& in)
      -> Map<const Matrix<typename ElementTraits<T>::value_type,    
             Dynamic, 1>>
  {
    return {
      reinterpret_cast<typename ElementTraits<T>::pointer>(in.data()),
      static_cast<int64_t>(in.size())
    };
  }

  template <typename T, int N>
  auto vec(TensorView_<T, N>& in)
      -> Map<const Matrix<typename ElementTraits<T>::value_type,    
             Dynamic, 1>>
  {
    return {
      reinterpret_cast<typename ElementTraits<T>::pointer>(in.data()),
      static_cast<int64_t>(in.size())
    };
  }

  template <typename T, int N>
  auto reshape_2d(TensorView_<T, N>& in, const Vector2i& shape)
      -> TensorView_<T, 2>
  {
    return {in.data(), shape};
  }

  template <typename T, int N>
  auto im2col(const TensorView_<T, N>& x, const Matrix<int, N, 1>& kernel_sizes)
      -> Tensor_<T, 2>
  {
    const auto num_rows = std::accumulate(
        x.sizes().data(),
        x.sizes().data() + x.sizes().size(), 1,
        std::multiplies<int>());
    const auto num_cols = std::accumulate(
        kernel_sizes.data(),
        kernel_sizes.data() + kernel_sizes.size(), 1,
        std::multiplies<int>());

    auto phi_x = Tensor_<T, 2>{num_rows, num_cols};

    const Matrix<T, N, 1> radius = kernel_sizes / 2;
    for (auto c = x.begin_array(); !c.end(); ++c)
    {
      auto s = c.coords() - radius;
      auto e = c.coords() + radius;

      phi_x.matrix().row(c) = vec(patch(x, s, e)).transpose();
    }

    return phi_x;
  }

  template <typename T, int N>
  void gemm_convolve(TensorView_<T, N>& kx,  //
                     const TensorView_<T, N>& x, const TensorView_<T, N>& k)
  {
    auto phi_x = im2col(x, k.sizes());
    kx.array() = reshape(phi_x.matrix() * k.vector(), x.sizes());
  }

  template <typename T>
  void gemm_batch_convolve(ImageView<T, 4>& fg,  //
                           const ImageView<T, 3>& f, const ImageView<T, 4>& g)
  {
    if (f.sizes().cwiseMax(g.sizes()) == f.sizes())
    {
      fg.matrix() =
          im2col(f, g.sizes()) *
          reshape_2d(g, {g.size(0), g.size(1) * g.size(2) * g.size(2)});
    }
    else
    {
      fg.matrix() =
          im2col(g, f.sizes()) *
          reshape_2d(f, {f.size(0), f.size(1) * f.size(2) * f.size(2)});
    }
  }

} /* namespace Sara */
} /* namespace DO */
