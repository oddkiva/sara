#pragma once

#include <DO/Sara/Core/ArrayIterators/ArrayIterators.hpp>
#include <DO/Sara/Core/Image.hpp>
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
    auto at(const MultiArrayView<T, N, O>& f, const Matrix<int, N, 1>& x) const
        -> T
    {
      auto y = x;

      // First pass.
      // Find the equivalent coordinate between [-2 * li, 2 * li[.
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

        else if (-li <= y[i] && y[i] < 0)
          y[i] = -y[i] - 1;

        else if (y[i] < -li)
          y[i] = y[i] + 2 * li;
      }

      return f(y);
    }
  };


  template <typename ArrayView, typename Padding>
  struct InfiniteMultiArrayView
  {
    using vector_type = typename ArrayView::vector_type;
    using value_type = typename ArrayView::value_type;

    InfiniteMultiArrayView(ArrayView&& f)
      : f{std::forward(f)}
    {
    }

    auto operator()(const vector_type& x) const -> value_type
    {
      return pad.ad(f, x);
    }

    ArrayView&& f;
    Padding pad;
  };

  template <typename ArrayView, typename Padding>
  inline auto make_infinite(ArrayView&& f, Padding pad)
      -> InfiniteMultiArrayView<ArrayView, Padding>
  {
    return {f};
  }

  template <typename ArrayView>
  class CoordsIterator
  {
  public:
    using self_type = CoordsIterator;
    using vector_type = typename ArrayView::vector_type;
    using value_type = typename ArrayView::value_type;
    using incrementer = PositionIncrementer<ArrayView::StorageOrder>;
    using decrementer = PositionDecrementer<ArrayView::StorageOrder>;

  public:
    CoordsIterator(const vector_type& begin, const vector_type& end)
      : _p{begin}
      , _begin{begin}
      , _end{end}
      , _stop{false}
    {
    }

    //! Dereferencing operator.
    inline const vector_type& operator*() const
    {
      return _p;
    }

    //! Referencing operator.
    inline const vector_type* operator->() const
    {
      return &_p;
    }

    //! Prefix increment operator.
    inline self_type& operator++()
    {
      incrementer::apply(_p, _stop, _begin, _end);
      return *this;
    }

    //! Prefix decrement operator.
    inline self_type& operator--()
    {
      decrementer::apply(_p, _stop, _begin, _end);
      return *this;
    }

    //! Postfix increment operator.
    inline self_type operator++(int)
    {
      self_type old{*this};
      operator++();
      return old;
    }

    //! Postfix increment operator.
    inline self_type operator--(int)
    {
      self_type old{*this};
      operator--();
      return old;
    }

    //! Arithmetic operator (slow).
    inline void operator+=(const vector_type& offset)
    {
      vector_type pos{_p + offset};
      _p = pos;
    }

    //! Arithmetic operator (slow).
    inline void operator-=(const vector_type& offset)
    {
      operator+=(-offset);
    }

    inline bool end() const {
      return _stop;
    }

  private:
    vector_type _p;
    vector_type _begin;
    vector_type _end;
    bool _stop;
  };


} /* namespace Sara */
} /* namespace DO */
