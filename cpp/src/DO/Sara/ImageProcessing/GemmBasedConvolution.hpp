#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Image/Operations.hpp>


namespace DO { namespace Sara {

  namespace runtime {

    template <typename T>
    class TensorView
    {
    public:
      DynamicTensorView() = default;

      DynamicTensorView& reshape(const std::vector<std::size_t>& shape);

    public:
      T* data{nullptr};
      std::vector<std::size_t> shape;
      std::vector<std::size_t> strides;
    };

    template <typename T>
    class Tensor
    {
    public:
      std::unique_ptr<T[]> data;
      std::vector<std::size_t> shape;
      std::vector<std::size_t> strides;
    };

  } /* namespace runtime */


  template <typename T>
  auto patch(const ImageView<T, 3>& in, const Vector3i& beg,
             const Vector3i& end) -> Image<T, 3>
  {
    return safe_crop(in, beg, end);
  }

  template <typename T, int N>
  auto reshape_2d(ImageView<T, N>& in, const Vector2i& shape)
      -> ImageView<T, 2>
  {
    return {in.data(), shape};
  }

  template <typename T, int N>
  auto im2col(const ImageView<T, N>& in, const Matrix<T, N, 1>& kernel_sizes)
      -> Matrix<T, Dynamic, Dynamic>
  {
    const auto num_rows = std::accumulate(
        in.sizes().data(), in.sizes().data() + in.sizes().size(), 1);
    const auto num_cols = std::accumulate(
        kernel_sizes.data(), kernel_sizes.data() + kernel_sizes.size(), 1);

    auto phi_x = Matrix<T, Dynamic, Dynamic>{num_rows, num_cols};

    const Matrix<T, N, 1> radius = kernel_sizes / 2;
    for (auto c = in.begin_array(); !c; ++c)
    {
      auto s = c.coords() - radius;
      auto e = c.coords() + radius;

      auto p = patch(in, s, e);
      phi_x.col(c) = flatten(patch(in, s, e));
    }

    return phi_x;
  }

  template <typename T, int N>
  auto im2col(const ImageView<T, N + 1>& in,
              const Matrix<T, N, 1>& kernel_sizes)
      -> Matrix<T, Dynamic, Dynamic>
  {
    Matrix<int, N, 1> sizes;
    sizes << in.size(0) * in.size(1), in.sizes().tail(in.sizes().size() - 2);
    return im2col(reshape(in, sizes), kernel_sizes);
  }

  template <typename T>
  void gemm_convolve(ImageView<T, 4>& kx,  //
                     const ImageView<T, 3>& x, const ImageView<T, 3>& k)
  {
    auto phi_x = im2col(x, k.sizes());
    kx.array() = reshape(phi_x.matrix() * vec(k), x.sizes());
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
