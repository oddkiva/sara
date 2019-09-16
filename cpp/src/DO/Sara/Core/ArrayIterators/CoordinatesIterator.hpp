// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/ArrayIterators/Utilities.hpp>


namespace DO { namespace Sara {

  template <typename ArrayView>
  class CoordinatesIterator
  {
  public:
    using self_type = CoordinatesIterator;
    using vector_type = typename ArrayView::vector_type;
    using value_type = typename ArrayView::value_type;
    using incrementer = PositionIncrementer<ArrayView::StorageOrder>;
    using decrementer = PositionDecrementer<ArrayView::StorageOrder>;

  public:
    CoordinatesIterator(const vector_type& begin, const vector_type& end)
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
    inline self_type operator+=(const vector_type& offset)
    {
      vector_type pos{_p + offset};
      _p = pos;
      return *this;
    }

    //! Arithmetic operator (slow).
    inline self_type operator-=(const vector_type& offset)
    {
      return operator+=(-offset);
    }

    inline bool end() const
    {
      return _stop;
    }

  private:
    vector_type _p;
    vector_type _begin;
    vector_type _end;
    bool _stop;
  };


  template <typename ArrayView>
  class SteppedCoordinatesIterator
  {
  public:
    using self_type = SteppedCoordinatesIterator;
    using vector_type = typename ArrayView::vector_type;
    using value_type = typename ArrayView::value_type;
    using incrementer = PositionIncrementer<ArrayView::StorageOrder>;
    using decrementer = PositionDecrementer<ArrayView::StorageOrder>;

  public:
    SteppedCoordinatesIterator(const vector_type& begin, const vector_type& end,
                               const vector_type& steps)
      : _p{begin}
      , _begin{begin}
      , _end{end}
      , _steps{steps}
      , _stop{false}
    {
    }

    //! @brief Dereferencing operator.
    inline const vector_type& operator*() const
    {
      return _p;
    }

    //! @brief Referencing operator.
    inline const vector_type* operator->() const
    {
      return &_p;
    }

    //! @brief Prefix increment operator.
    inline self_type& operator++()
    {
      incrementer::apply(_p, _stop, _begin, _end, _steps);
      return *this;
    }

    //! @brief Prefix decrement operator.
    inline self_type& operator--()
    {
      decrementer::apply(_p, _stop, _begin, _end, _steps);
      return *this;
    }

    //! @brief Postfix increment operator.
    inline self_type operator++(int)
    {
      self_type old{*this};
      operator++();
      return old;
    }

    //! @brief Postfix increment operator.
    inline self_type operator--(int)
    {
      self_type old{*this};
      operator--();
      return old;
    }

    //! @brief Arithmetic operator (slow).
    inline self_type operator+=(const vector_type& offset)
    {
      vector_type pos{_p + offset.cwiseProduct(_steps)};
      _p = pos;
      return *this;
    }

    //! @brief Arithmetic operator (slow).
    inline self_type operator-=(const vector_type& offset)
    {
      return operator+=(-offset);
    }

    inline bool end() const
    {
      return _stop;
    }

    //! @brief Return the sizes of the stepped array.
    inline vector_type stepped_subarray_sizes() const
    {
      auto sizes = vector_type{};
      for (int i = 0; i < sizes.size(); ++i)
      {
        const auto modulo = (_end[i] - _begin[i]) % _steps[i];
        sizes[i] = (_end[i] - _begin[i]) / _steps[i] + int(modulo != 0);
      }
      return sizes;
    }

  private:
    vector_type _p;
    vector_type _begin;
    vector_type _end;
    vector_type _steps;
    bool _stop;
  };

} /* namespace Sara */
} /* namespace DO */
