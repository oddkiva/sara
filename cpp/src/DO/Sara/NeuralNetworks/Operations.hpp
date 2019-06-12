#pragma once

#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>
#include <DO/Sara/NeuralNetworks/Variables.hpp>


namespace DO { namespace Sara {

  template <typename Op>
  struct Operation
  {
  };


  template <typename T, typename Padding>
  struct Conv2D : Operation<Conv2D<T, Padding>>
  {
    using in_type = Variable<Tensor_<T, 4>>;
    using out_type = Variable<Tensor_<T, 4>>;

    void eval(out_type& y, const in_type& x)
    {
      y = gemm_convolve(x, w, padding, strides, offset) + b;
    }

    //! Learnable parameters.
    Variable<Tensor_<T, 4>> w;
    Variable<Tensor_<T, 4>> b;

    //! Other parameters.
    Padding padding;
    Matrix<int, 4, 1> strides;
    Matrix<int, 4, 1> offset;
  };


  template <typename T>
  struct Dense : public Operation<Dense<T>>
  {
    using in_type = Variable<Tensor_<T, 4>>;
    using out_type = Variable<Tensor_<T, 4>>;

    void eval(out_type& y, const in_type& x)
    {
      y = w * x + b;
    }

    //! Learnable parameters.
    Variable<Tensor_<T, 4>> w;
    Variable<Tensor_<T, 4>> b;
  };

} /* namespace Sara */
} /* namespace DO */
