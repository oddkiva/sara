#pragma once

#include <DO/Sara/Core/ArrayIterators/ArrayIterators.hpp>
#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>
#include <DO/Sara/Core/Pixel/PixelTraits.hpp>


namespace DO { namespace Sara {

  template <typename T>
  class ConstantPadding
  {
  public:
    ConstantPadding() = default;

    template <int N, int O>
    auto at(MultiArrayView<T, N, O>& f, const Matrix<int, N, 1>& x) const
        -> const T&
    {
      if (x.minCoeff() < 0 || (x - f.sizes()).minCoeff() >= 0)
        return value;

      return f(x);
    }

  private:
    T value{PixelTraits<T>::min()};
  };

  template <typename DF, typename F>
  class NeumannPadding
  {
  public:
    NeumannPadding() = default;

    template <int N, int O>
    auto at(MultiArrayView<F, N, O>& f, const Matrix<int, N, 1>& x) const
        -> const F&
    {
      if (x.minCoeff() < 0)
        return f(x) + _df_x * x;

      if ((x - f.sizes()).minCoeff() >= 0)
        f(x) + _df_x * (x - f.sizes());

      return f(x);
    }

  private:
    DF _df_x;
  };

  class PeriodicPadding
  {
  public:
    PeriodicPadding() = default;

    template <typename T, int N, int O>
    auto at(MultiArrayView<T, N, O>& f, const Matrix<int, N, 1>& x) const
        -> const T&
    {
      auto y = x;

      // First pass.
      // Find the equivalent coordinate between ]-2 * li, 2 * li[.
      for (auto i = 0; i < N; ++i)
      {
        const auto li = f.size(i);

        if (x[i] >= 0)
          y[i] = x[i] % (2*li);
        else
          y[i] = -(std::abs(x[i]) % (2 * li));
      }

      // Second pass.
      for (auto i = 0; i < N; ++i)
      {
        const auto li = f.size(i);

        if (0 <= y[i] && y[i] < li)
          continue;

        else if (y[i] >= li)
          y[i] = 2 * li - y[i] - 1;

        else if (-li < y[i] && y[i] < 0)
          y[i] = -y[i];

        else if (y[i] <= -li)
          y[i] = y[i] + 2 * li;
      }

      return f(y);
    }
  };


  template <typename T, int N, int O>
  class InfiniteMultiArrayView
  {

  };

  template <typename PaddingCondition, typename ArrayView>
  class InfiniteArrayIterator
  {
  public:
    using vector_type = typename ArrayView::vector_type;
    using incrementer = PositionIncrementer<ArrayView::StorageOrder>;
    using decrementer = PositionDecrementer<ArrayView::StorageOrder>;

  private:
    ArrayView&& _f;
    vector_type _x;
  };


} /* namespace Sara */
} /* namespace DO */
