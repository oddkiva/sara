#pragma once

#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>
#include <DO/Sara/NeuralNetworks/Variables.hpp>

#include <memory>


namespace DO { namespace Sara {

  // A calculator.
  template <typename Padding = PeriodicPadding>
  struct Conv2D
  {
    Conv2D() = default;

    auto operator()(const Tensor_<float, 4>& x) const
    {
      constexpr auto N = 4;
      const Vector4i strides{1, w.size(0), 1, 1};
      const Vector4i offset{0, 1, 0, 0};

      // The gradient of the convolution is precomputed.
      auto grad_y = nabla_w(x);

      // Determine the sizes of the kernel.
      Matrix<int, N, 1> k_sizes;
      k_sizes << w.sizes()[N - 1], w.sizes().head(N - 1);
      const auto krows = std::accumulate(k_sizes.data() + 1, k_sizes.data() + N,
                                         1, std::multiplies<int>());
      const auto kcols = k_sizes[0];
      auto kt = w.reshape(Vector2i{krows, kcols});

      // Determine the sizes of the convolutional output.
      auto y_sizes =
          x.begin_stepped_subarray(Vector4i::Zero(), x.sizes(), strides)
              .stepped_subarray_sizes();
      y_sizes[1] = kcols;

      // Perform the convolution.
      auto y = Tensor_<float, 4>{y_sizes};
      y.colmajor_view()                                                  //
          .reshape(Vector2i{grad_y.matrix().rows(), kt.matrix().cols()})  //
          .matrix() = grad_y.matrix() * kt.matrix();

      return std::make_tuple(y, grad_y);
    }

    auto nabla_w(const Tensor_<float, 4>& x) const
    {
      const Vector4i strides{1, w.size(0), 1, 1};
      const Vector4i offset{0, 1, 0, 0};
      constexpr auto N = 4;

      // Determine the sizes of the kernel.
      Matrix<int, N, 1> k_sizes;
      k_sizes << w.sizes()[N - 1], w.sizes().head(N - 1);

      // Calculate the feature maps for each Nd-pixel.
      k_sizes[0] = 1;
      auto phi_x = im2col(x, k_sizes, pad, strides, offset);

      return phi_x;
    }

    Padding pad;
    Tensor_<float, 4> w;
    Tensor_<float, 4> b;
  };


  struct Dense
  {
    Dense() = default;

    Tensor_<float, 4> w;
    Tensor_<float, 4> b;
  };


  template <typename F>
  struct Derivative;

  //template <>
  //struct Derivative<Conv2D> : public Operation<Derivative<Conv2D>>
  //{
  //  std::shared_ptr<Tensor_<float, 4>> _phi_x;
  //};

  //template <>
  //struct Derivative<Dense> : public Operation<Derivative<Dense>>
  //{
  //  std::shared_ptr<Tensor_<float, 4>> _x;
  //};


  class CrossEntropyLoss
  {
  };


} /* namespace Sara */
} /* namespace DO */
