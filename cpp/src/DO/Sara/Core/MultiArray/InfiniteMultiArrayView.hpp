// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/CoordinatesIterator.hpp>
#include <DO/Sara/Core/MultiArray.hpp>
#include <DO/Sara/Core/Pixel/PixelTraits.hpp>


namespace DO { namespace Sara {

  template <typename T>
  class ConstantPadding
  {
  public:
    ConstantPadding(T value)
      : _value{value}
    {
    }

    template <int N, int O>
    auto at(const MultiArrayView<T, N, O>& f, const Matrix<int, N, 1>& x) const
        -> T
    {
      if (x.minCoeff() < 0 || (x - f.sizes()).maxCoeff() >= 0)
        return _value;

      return f(x);
    }

  private:
    T _value;
  };

  template <typename T>
  auto make_constant_padding(T&& value) -> ConstantPadding<T>
  {
    return {value};
  }

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
        static_assert(std::is_same<decltype(li), const int>::value, "");

        if (x[i] >= 0)
          y[i] = x[i] % (2 * li);
        else
          y[i] = -(-x[i] % (2 * li));
      }

      // Second pass.
      // Find the equivalent coordinate between [0, li[.
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


  template <typename ArrayView>
  class InfiniteMultiArrayViewIterator
  {
  public:
    using self_type = InfiniteMultiArrayViewIterator;
    using vector_type = typename ArrayView::vector_type;
    using value_type = typename ArrayView::value_type;

  public:
    inline InfiniteMultiArrayViewIterator(const ArrayView& f,    //
                                          const vector_type& a,  //
                                          const vector_type& b)
      : _f{f}
      , _x{a, b}
    {
    }

    //! Dereferencing operator.
    inline value_type operator*() const
    {
      return _f(*_x);
    }

    //! Prefix increment operator.
    inline self_type& operator++()
    {
      ++_x;
      return *this;
    }

    //! Prefix decrement operator.
    inline self_type& operator--()
    {
      --_x;
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
    inline self_type operator+=(const vector_type& offset)
    {
      _x += offset;
      return *this;
    }

    //! Arithmetic operator (slow).
    inline self_type operator-=(const vector_type& offset)
    {
      _x +=(-offset);
      return *this;
    }

    inline bool end() const
    {
      return _x.end();
    }


    inline auto position() const -> const vector_type&
    {
      return *_x;
    }

  private:
    const ArrayView& _f;
    CoordinatesIterator<ArrayView> _x;
  };


  template <typename ArrayView, typename Padding>
  class InfiniteMultiArrayView
  {
  public:
    using vector_type = typename ArrayView::vector_type;
    using value_type = typename ArrayView::value_type;

    enum { StorageOrder = ArrayView::StorageOrder };

    InfiniteMultiArrayView(const ArrayView& f, const Padding& pad)
      : _f(f)
      , _pad{pad}
    {
    }

    auto operator()(const vector_type& x) const -> value_type
    {
      return _pad.at(_f, x);
    }

    auto begin_subarray(const vector_type& a, const vector_type& b) const
        -> InfiniteMultiArrayViewIterator<InfiniteMultiArrayView>
    {
      return {*this, a, b};
    }

  private:
    const ArrayView& _f;
    Padding _pad;
  };

  template <typename ArrayView, typename Padding>
  inline auto make_infinite(const ArrayView& f, const Padding& pad)
      -> InfiniteMultiArrayView<ArrayView, Padding>
  {
    return {f, pad};
  }

} /* namespace Sara */
} /* namespace DO */
