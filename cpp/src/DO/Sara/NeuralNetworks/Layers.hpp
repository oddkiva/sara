#pragma once

#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>
#include <DO/Sara/NeuralNetworks/Variables.hpp>

#include <memory>


namespace DO { namespace Sara {

  template <typename Op>
  struct Layer
  {
  };


  class Conv2D : public Layer<Conv2D>
  {
  public:
    Conv2D() = default;

  private:
    Variable<Tensor_<float, 4>> _w;
    Variable<Tensor_<float, 4>> _b;

    std::shared_ptr<Tensor_<float, 4>> _x;
    std::shared_ptr<Tensor_<float, 4>> _phi_x;

    std::shared_ptr<Tensor_<float, 4>> _y;
  };


  class Dense : public Layer<Dense>
  {
  public:
    Dense() = default;


  private:
    Tensor_<float, 4> _w;
    Tensor_<float, 4> _b;

    std::shared_ptr<Tensor_<float, 4>> _x;
    std::shared_ptr<Tensor_<float, 4>> _y;
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
