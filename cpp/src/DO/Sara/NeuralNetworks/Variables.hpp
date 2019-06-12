#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO { namespace Sara {

  template <typename X>
  struct Expr
  {
  };


  template <typename T>
  struct Variable : public Expr<Variable<T>>
  {
    T value;
    std::string name;

    inline operator T&()
    {
      return value;
    }

    inline operator const T&()
    {
      return value;
    }
  };


  template <typename F, typename X>
  struct FuncXpr : public Expr<FuncXpr<F, X>>
  {
  };

  template <typename X, typename Y>
  struct PlusXpr : public Expr<PlusXpr<X, Y>>
  {
  };

} /* namespace Sara */
} /* namespace DO */
