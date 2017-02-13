// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
//! @brief Implementation of N-dimensional iterators.

#pragma once

#include <DO/Sara/Core/ArrayIterators/Utilities.hpp>


namespace DO { namespace Sara {

  //! @ingroup Core
  //! @defgroup MultiArrayIterators ND-array iterator classes.
  //! @{


  //! @{
  //! MultiArray iterator classes.
  template <bool IsConst, typename T, int N, int StorageOrder>
  class ArrayIteratorBase;

  template <bool IsConst, typename T, int N, int StorageOrder>
  class ArrayIterator;

  template <bool IsConst, typename T, int N, int StorageOrder>
  class SubarrayIterator;

  template <bool IsConst, typename T, int Axis, int N>
  class AxisIterator;
  //! @}


  //! @{
  //! Convenient typedefs.
#define ITERATOR_BASE_TYPE(IsConst)                                            \
  std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t,            \
                typename Meta::Choose<IsConst, const T *, T *>::Type,          \
                typename Meta::Choose<IsConst, const T&, T&>::Type>

#define TYPEDEF_ITERATOR_TYPES(IteratorType)                                   \
  using value_type = typename base_type::value_type;                           \
  using difference_type = typename base_type::difference_type;                 \
  using pointer = typename base_type::pointer;                                 \
  using reference = typename base_type::reference;                             \
  using iterator_category = typename base_type::iterator_category
  //! @}


  //! @brief Axis iterator class for N-dimensional arrays.
  template <bool IsConst, typename T, int Axis, int N>
  class AxisIterator : public ITERATOR_BASE_TYPE(IsConst)
  {
    static_assert(Axis >= 0 && Axis < N,
                  "Axis must be nonnegative and less than N");

    // Friend classes.
    template <bool, typename, int, int>
    friend class AxisIterator;
    template <bool, typename, int, int>
    friend class ArrayIteratorBase;

    // Private typedefs.
    using base_type = ITERATOR_BASE_TYPE(IsConst);
    using self_type = AxisIterator;

  public: /* STL-like typedefs */
    TYPEDEF_ITERATOR_TYPES(base_type);
    using vector_type = Matrix<int, N, 1>;

  public: /* interface */
    //! Constructor.
    inline AxisIterator(pointer& ptr, vector_type& pos,
                        const vector_type& strides, const vector_type& sizes)
      : _cur_ptr(ptr)
      , _cur_pos(pos)
      , _strides(strides)
      , _sizes(sizes)
    {
    }

  public: /* dereferencing, access functions. */
    //! Dereferencing operator.
    inline reference operator*() const
    {
      return *_cur_ptr;
    }

    //! Dereferencing operator.
    inline pointer operator->() const
    {
      return _cur_ptr;
    }

    //! Access operator.
    inline reference operator[](int n) const
    {
      if (_cur_pos[Axis] + n < 0 || _cur_pos[Axis] + n >= _sizes[Axis])
        throw std::out_of_range{"Axis iterator is out of range"};
      return *(_cur_ptr + _strides[Axis] * n);
    }

  public: /* comparison functions. */
    //! Equality operator.
    template <bool IsConst2, int Axis2>
    inline bool operator==(const AxisIterator<IsConst2, T, Axis2, N>& rhs) const
    {
      return _cur_ptr == rhs._cur_ptr;
    }

    //! Equality operator.
    template <bool IsConst2, int StorageOrder>
    inline bool
    operator==(const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& rhs) const
    {
      return _cur_ptr == rhs._cur_ptr;
    }

    //! Equality operator.
    inline bool operator==(const T *ptr) const
    {
      return _cur_ptr == ptr;
    }

    //! Inequality operator.
    template <bool IsConst2, int Axis2>
    inline bool
    operator!=(const AxisIterator<IsConst2, T, Axis2, N>& other) const
    {
      return !operator==(other);
    }

    //! Inequality operator.
    template <bool IsConst2, int StorageOrder>
    inline bool operator!=(
        const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& other) const
    {
      return !operator==(other);
    }

    //! Inequality operator.
    inline bool operator!=(const T *ptr) const
    {
      return !operator==(ptr);
    }

  public: /* arithmetic operators. */
    //! Addition operator.
    inline void operator+=(int n)
    {
      if (_cur_pos[Axis] + n < 0 || _cur_pos[Axis] + n >= _sizes[Axis])
        throw std::out_of_range{"Axis iterator is out of range"};
      _cur_ptr += _strides[Axis] * n;
      _cur_pos[Axis] += n;
    }

    //! Subtraction operator.
    inline void operator-=(int n)
    {
      return operator+=(-n);
    }

  public: /* iterator functionalities. */
    //! Prefix increment operator.
    inline self_type& operator++()
    {
      operator+=(1);
      return *this;
    }

    //! Prefix decrement operator.
    inline self_type& operator--()
    {
      operator-=(1);
      return *this;
    }

    //! Postfix increment operator.
    inline self_type operator++(int)
    {
      AxisIterator old{*this};
      operator++();
      return old;
    }

    //! Postfix decrement operator.
    inline self_type operator--(int)
    {
      AxisIterator old{*this};
      operator--();
      return old;
    }

  private: /* data members. */
    pointer& _cur_ptr;            //!< current pointer.
    vector_type& _cur_pos;        //!< current coordinates.
    const vector_type& _strides;  //!< strides.
    const vector_type& _sizes;    //!< sizes.
  };


  //! @brief Multidimensional iterator base class.
  //! The 'ArrayIteratorBase' class is a heavy object. It is mostly useful
  //! for differential calculus. When possible, prefer using more elementary
  //! iterator instead.
  template <bool IsConst, typename T, int N, int StorageOrder>
  class ArrayIteratorBase : public ITERATOR_BASE_TYPE(IsConst)
  {
    using base_type = ITERATOR_BASE_TYPE(IsConst);
    template <bool, typename, int, int>
    friend class AxisIterator;
    template <bool, typename, int, int>
    friend class ArrayIteratorBase;

  public: /* typedefs */
    TYPEDEF_ITERATOR_TYPES(base_type);
    using self_type = ArrayIteratorBase;
    using vector_type = Matrix<int, N, 1>;
    using x_iterator = AxisIterator<IsConst, value_type, 0, N>;
    using y_iterator = AxisIterator<IsConst, value_type, 1, N>;
    using z_iterator = AxisIterator<IsConst, value_type, 2, N>;

  public: /* interface */
    //! @brief Constructor
    inline ArrayIteratorBase(bool stop, pointer ptr, const vector_type& pos,
                             const vector_type& strides,
                             const vector_type& sizes)
      : base_type{}
      , _stop{stop}
      , _cur_ptr(ptr)
      , _cur_pos(pos)
      , _strides(strides)
      , _sizes(sizes)
    {
    }

  public: /* dereferencing functions. */
    //! Dereferencing operator.
    inline reference operator*() const
    {
      return *_cur_ptr;
    }

    //! Referencing operator.
    inline pointer operator->() const
    {
      return _cur_ptr;
    }

  public: /* comparison functions. */
    //! Equality operator.
    inline bool operator==(const self_type& other) const
    {
      return _cur_ptr == other._cur_ptr;
    }

    //! Equality operator.
    template <bool IsConst2>
    inline bool operator==(
        const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& other) const
    {
      return _cur_ptr == other._cur_ptr;
    }

    //! Equality operator.
    inline bool operator==(const T *ptr) const
    {
      return _cur_ptr == ptr;
    }

    //! Inequality operator.
    inline bool operator!=(const self_type& other) const
    {
      return !this->operator==(other);
    }

    //! Inequality operator.
    template <bool IsConst2>
    inline bool operator!=(
        const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& other) const
    {
      return !this->operator==(other);
    }

    //! Inequality operator.
    inline bool operator!=(const T* ptr) const
    {
      return !operator==(ptr);
    }

  public: /* special access operators. */
    //! Special access operators.
    inline reference operator()(const vector_type& offset) const
    {
      vector_type pos(_cur_pos + offset);
      if (pos.minCoeff() < 0 || (pos - _sizes).minCoeff() >= 0)
        throw std::out_of_range{"Range iterator out of range!"};
      return *(_cur_ptr + jump(offset, _strides));
    }

    //! Special access operator in 2D.
    inline reference operator()(int i, int j) const
    {
      static_assert(N == 2, "Data must be 2D");
      return operator()(vector_type(i, j));
    }

    //! Special access operator in 3D.
    inline reference operator()(int i, int j, int k) const
    {
      static_assert(N == 3, "Data must be 3D");
      return operator()(vector_type(i, j, k));
    }

    //! Special access operator.
    inline reference delta(int axis_i, int step_i) const
    {
      return *(_cur_ptr + _strides[axis_i] * step_i);
    }

    //! Special access operator (mostly for the Hessian matrix).
    inline reference delta(int axis_i, int step_i, int axis_j, int step_j) const
    {
      return *(_cur_ptr + _strides[axis_i] * step_i +
               _strides[axis_j] * step_j);
    }

    //! Special access operator (mostly for the Hessian matrix).
    template <int I, int J>
    inline reference delta(int step_i, int step_j) const
    {
      static_assert(I >= 0 && I < N && J >= 0 && J < N,
                    "I and J must between 0 and N");
      return *(_cur_ptr + _strides[I] * step_i + _strides[J] * step_j);
    }

    //! Axis iterator getter.
    //! The axes matches with the Cartesian view if the data is stored in a
    //! row major fashion.
    template <int Axis>
    inline AxisIterator<IsConst, T, Axis, N> axis()
    {
      return AxisIterator<IsConst, T, Axis, N>(_cur_ptr, _cur_pos, _strides,
                                               _sizes);
    }

    //! X-axis iterator getter.
    inline x_iterator x()
    {
      return axis<0>();
    }

    //! Y-axis iterator getter.
    inline y_iterator y()
    {
      return axis<1>();
    }

    //! Z-Axis iterator getter.
    inline z_iterator z()
    {
      return axis<2>();
    }

  public: /* additional features. */
    //! Return the current position in the the array.
    inline const vector_type& position() const
    {
      return _cur_pos;
    }

    //! Return the array strides.
    inline const vector_type& strides() const
    {
      return _strides;
    }

    //! Return the array sizes.
    inline const vector_type& sizes() const
    {
      return _sizes;
    }

    //! Check if the iteration process is finished.
    inline bool end() const
    {
      return _stop;
    }

  protected: /* data members */
    bool _stop;
    pointer _cur_ptr;             //!< current pointer.
    vector_type _cur_pos;         //!< current coordinates.
    const vector_type& _strides;  //!< strides.
    const vector_type& _sizes;    //!< sizes.
  };


  //! @brief Multidimensional iterator class.
  //! The 'ArrayIterator' class is a heavy object. It is mostly useful for
  //! differential calculus. When possible, prefer using more elementary
  //! iterator instead.
  template <bool IsConst, typename T, int N, int StorageOrder>
  class ArrayIterator : public ArrayIteratorBase<IsConst, T, N, StorageOrder>
  {
    static_assert(N >= 0, "Number of dimensions must be nonnegative");
    using base_type = ArrayIteratorBase<IsConst, T, N, StorageOrder>;
    using self_type = ArrayIterator;
    using incrementer = PositionIncrementer<StorageOrder>;
    using decrementer = PositionDecrementer<StorageOrder>;

    using base_type::_cur_pos;
    using base_type::_cur_ptr;
    using base_type::_stop;
    using base_type::_sizes;
    using base_type::_strides;

  public:
    TYPEDEF_ITERATOR_TYPES(base_type);
    using vector_type = Matrix<int, N, 1>;
    using pointer_reference = pointer&;
    using vector_reference = vector_type&;

  public:
    inline ArrayIterator(bool stop, const pointer ptr, const vector_type& pos,
                         const vector_type& sizes, const vector_type& strides)
      : base_type{stop, ptr, pos, strides, sizes}
    {
    }

  public: /* iteration methods. */
    //! Prefix increment operator.
    inline self_type& operator++()
    {
      ++_cur_ptr;
      incrementer::apply(_cur_pos, _stop, _sizes);
      return *this;
    }

    //! Prefix decrement operator.
    inline self_type& operator--()
    {
      --_cur_ptr;
      decrementer::apply(_cur_pos, _stop, _sizes);
      return *this;
    }

    //! Postfix increment operator.
    inline self_type operator++(int)
    {
      self_type old{*this};
      operator++();
      return old;
    }

    //! Postfix decrement operator.
    inline self_type operator--(int)
    {
      self_type old{*this};
      operator--();
      return old;
    }

    //! Addition operator (slow).
    inline void operator+=(const vector_type& offset)
    {
      vector_type pos{_cur_pos + offset};
      if (pos.minCoeff() < 0 || (pos - _sizes).minCoeff() >= 0)
        throw std::out_of_range{"Range iterator out of range!"};
      _cur_ptr += jump(offset, _strides);
      _cur_pos = pos;
    }

    //! Subtraction operator (slow).
    inline void operator-=(const vector_type& offset)
    {
      operator+=(-offset);
    }
  };


  //! @brief Multidimensional iterator base class.
  //! The 'SubarrayIterator' class is a heavy object. It is mostly useful for
  //! differential calculus. When possible, prefer using more elementary
  //! iterator instead.
  template <bool IsConst, typename T, int N, int StorageOrder = ColMajor>
  class SubarrayIterator : public ArrayIteratorBase<IsConst, T, N, StorageOrder>
  {
    static_assert(N >= 0, "Number of dimensions must be nonnegative");

    using base_type = ArrayIteratorBase<IsConst, T, N, StorageOrder>;
    using self_type = SubarrayIterator;
    using incrementer = PositionIncrementer<StorageOrder>;
    using decrementer = PositionDecrementer<StorageOrder>;

    using base_type::_stop;
    using base_type::_cur_pos;
    using base_type::_cur_ptr;
    using base_type::_strides;

  public: /* typedefs. */
    TYPEDEF_ITERATOR_TYPES(base_type);
    using coords_type = Matrix<int, N, 1>;
    using vector_type = Matrix<int, N, 1>;

  public: /* constructors */
    //! Constructor
    inline SubarrayIterator(bool stop, pointer ptr,
                            const vector_type& begin_pos,
                            const vector_type& end_pos,
                            const vector_type& strides,
                            const vector_type& sizes)
      : base_type{stop, ptr + jump(begin_pos, strides), begin_pos, strides,
                  sizes}
      , _begin{ptr}
      , _begin_pos{begin_pos}
      , _end_pos{end_pos}
    {
    }

  public: /* iterator functionalities. */
    //! Prefix increment operator.
    inline self_type& operator++()
    {
      incrementer::apply(_cur_pos, _stop, _begin_pos, _end_pos);
      _cur_ptr = _begin + jump(_cur_pos, _strides);
      return *this;
    }

    //! Prefix decrement operator.
    inline self_type& operator--()
    {
      decrementer::apply(_cur_pos, _stop, _begin_pos, _end_pos);
      _cur_ptr = _begin + jump(_cur_pos, _strides);
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
      vector_type pos{_cur_pos + offset};
      if ((pos - _begin_pos).minCoeff() < 0 || (pos - _end_pos).minCoeff() >= 0)
      {
        std::ostringstream msg;
        msg << "Subrange iterator out of range: pos = " << offset.transpose();
        throw std::out_of_range{msg.str()};
      }
      _cur_pos = pos;
      _cur_ptr += jump(offset, _strides);
    }

    //! Arithmetic operator (slow).
    inline void operator-=(const vector_type& offset)
    {
      operator+=(-offset);
    }

  protected: /* data members. */
    pointer _begin;
    vector_type _begin_pos;
    vector_type _end_pos;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
