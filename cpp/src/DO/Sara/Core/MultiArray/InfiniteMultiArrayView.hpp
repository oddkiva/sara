// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/ArrayIterators/CoordinatesIterator.hpp>
#include <DO/Sara/Core/MultiArray/Padding.hpp>


namespace DO { namespace Sara {

  template <typename ArrayView>
  class InfiniteArrayIterator
  {
  public:
    using self_type = InfiniteArrayIterator;
    using vector_type = typename ArrayView::vector_type;
    using value_type = typename ArrayView::value_type;

  public:
    inline InfiniteArrayIterator(const ArrayView& f,    //
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
        -> InfiniteArrayIterator<InfiniteMultiArrayView>
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
