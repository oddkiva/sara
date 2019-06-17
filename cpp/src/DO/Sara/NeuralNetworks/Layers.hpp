#pragma once

#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>
#include <DO/Sara/NeuralNetworks/Variables.hpp>

#include <memory>


namespace DO { namespace Sara {

  class Conv2D
  {
  public:
    Conv2D() = default;

  private:
    Variable<TensorView_<float, 4>> _w;
    Variable<TensorView_<float, 4>> _b;
  };


  class Dense
  {
  public:
    Dense() = default;

  private:
    Variable<TensorView_<float, 4>> _w;
    Variable<TensorView_<float, 4>> _b;
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
