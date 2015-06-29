// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
//! \brief Implementation of N-dimensional iterators.

#ifndef DO_SARA_CORE_ARRAYITERATORS_MULTIARRAYITERATORS_HPP
#define DO_SARA_CORE_ARRAYITERATORS_MULTIARRAYITERATORS_HPP


#include <DO/Sara/Core/ArrayIterators/Utilities.hpp>


namespace DO { namespace Sara {

  //! \ingroup Core
  //! \defgroup MultiArrayIterators ND-array iterator classes.
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
#define ITERATOR_BASE_TYPE(IsConst)                       \
  std::iterator<                                          \
    std::random_access_iterator_tag, T, std::ptrdiff_t,   \
    typename Meta::Choose<IsConst, const T *, T *>::Type, \
    typename Meta::Choose<IsConst, const T&, T&>::Type    \
  >

#define TYPEDEF_ITERATOR_TYPES(IteratorType)                      \
  using value_type = typename base_type::value_type;              \
  using difference_type = typename base_type::difference_type;    \
  using pointer = typename base_type::pointer;                    \
  using reference = typename base_type::reference;                \
  using iterator_category = typename base_type::iterator_category
  //! @}


  //! \brief Axis iterator class for N-dimensional arrays.
  template <bool IsConst, typename T, int Axis, int N>
  class AxisIterator : public ITERATOR_BASE_TYPE(IsConst)
  {
    static_assert(
      Axis >= 0 && Axis < N,
      "Axis must be nonnegative and less than N");

    // Friend classes.
    template <bool, typename, int, int> friend class AxisIterator;
    template <bool, typename, int, int> friend class ArrayIteratorBase;

    // Private typedefs.
    using base_type = ITERATOR_BASE_TYPE(IsConst);
    using self_type = AxisIterator;

  public: /* STL-like typedefs */
    TYPEDEF_ITERATOR_TYPES(base_type);
    using vector_type = Matrix<int, N, 1>;

  public: /* interface */
    //! Constructor.
    inline AxisIterator(pointer& ptr,
                        vector_type& pos,
                        const vector_type& strides,
                        const vector_type& sizes)
      : cur_ptr_(ptr)
      , cur_pos_(pos)
      , strides_(strides)
      , sizes_(sizes)
    {
    }

  public: /* dereferencing, access functions. */
    //! Dereferencing operator.
    inline reference operator*() const
    {
      return *cur_ptr_;
    }

    //! Dereferencing operator.
    inline pointer operator->() const
    {
      return cur_ptr_;
    }

    //! Access operator.
    inline reference operator[](int n) const
    {
      if (cur_pos_[Axis]+n < 0  || cur_pos_[Axis]+n >= sizes_[Axis])
        throw std::out_of_range("Axis iterator is out of range");
      return *(cur_ptr_+strides_[Axis]*n);
    }

  public: /* comparison functions. */
    //! Equality operator.
    template <bool IsConst2, int Axis2>
    inline bool operator==(const AxisIterator<IsConst2, T, Axis2, N>& rhs) const
    {
      return cur_ptr_ == rhs.cur_ptr_;
    }

    //! Equality operator.
    template <bool IsConst2, int StorageOrder>
    inline bool operator==(const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& rhs) const
    {
      return cur_ptr_ == rhs.cur_ptr_;
    }

    //! Equality operator.
    inline bool operator==(const T *ptr) const
    {
      return cur_ptr_ == ptr;
    }

    //! Inequality operator.
    template <bool IsConst2, int Axis2>
    inline bool operator!=(const AxisIterator<IsConst2, T, Axis2, N>& other) const
    {
      return !operator==(other);
    }

    //! Inequality operator.
    template <bool IsConst2, int StorageOrder>
    inline bool operator!=(const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& other) const
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
      if (cur_pos_[Axis]+n < 0  || cur_pos_[Axis]+n >= sizes_[Axis])
        throw std::out_of_range("Axis iterator is out of range");
      cur_ptr_ += strides_[Axis]*n;
      cur_pos_[Axis] += n;
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
      AxisIterator old{ *this };
      operator++();
      return old;
    }

    //! Postfix decrement operator.
    inline self_type operator--(int)
    {
      AxisIterator old{ *this };
      operator--();
      return old;
    }

  private: /* data members. */
    pointer& cur_ptr_;            //!< current pointer.
    vector_type& cur_pos_;        //!< current coordinates.
    const vector_type& strides_;  //!< strides.
    const vector_type& sizes_;    //!< sizes.
  };


  //! \brief Multidimensional iterator base class.
  //! The 'ArrayIteratorBase' class is a heavy object. It is mostly useful
  //! for differential calculus. When possible, prefer using more elementary
  //! iterator instead.
  template <bool IsConst, typename T, int N, int StorageOrder>
  class ArrayIteratorBase : public ITERATOR_BASE_TYPE(IsConst)
  {
    using base_type = ITERATOR_BASE_TYPE(IsConst);
    template <bool, typename, int, int> friend class AxisIterator;
    template <bool, typename, int, int> friend class ArrayIteratorBase;

  public: /* typedefs */
    TYPEDEF_ITERATOR_TYPES(base_type);
    using self_type = ArrayIteratorBase;
    using vector_type = Matrix<int, N, 1>;
    using x_iterator = AxisIterator<IsConst, value_type, 0, N>;
    using y_iterator = AxisIterator<IsConst, value_type, 1, N>;
    using z_iterator = AxisIterator<IsConst, value_type, 2, N>;

  public: /* interface */
    //! \brief Constructor
    inline ArrayIteratorBase(bool stop,
                             pointer ptr,
                             const vector_type& pos,
                             const vector_type& strides,
                             const vector_type& sizes)
      : base_type{}
      , stop_{stop}
      , cur_ptr_(ptr)
      , cur_pos_(pos)
      , strides_(strides)
      , sizes_(sizes)
    {
    }

  public: /* dereferencing functions. */
    //! Dereferencing operator.
    inline reference operator*() const
    {
      return *cur_ptr_;
    }

    //! Referencing operator.
    inline pointer operator->() const
    {
      return cur_ptr_;
    }

  public: /* comparison functions. */
    //! Equality operator.
    inline bool operator==(const self_type& other) const
    {
      return cur_ptr_ == other.cur_ptr_;
    }

    //! Equality operator.
    template <bool IsConst2>
    inline bool operator==(
      const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& other
    ) const
    {
      return cur_ptr_ == other.cur_ptr_;
    }

    //! Equality operator.
    inline bool operator==(const T *ptr) const
    {
      return cur_ptr_ == ptr;
    }

    //! Inequality operator.
    inline bool operator!=(const self_type& other) const
    {
      return !this->operator==(other);
    }

    //! Inequality operator.
    template <bool IsConst2>
    inline bool operator!=(
      const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& other
    ) const
    {
      return !this->operator==(other);
    }

    //! Inequality operator.
    inline bool operator!=(const T *ptr) const
    {
      return !operator==(ptr);
    }

  public: /* special access operators. */
    //! Special access operators.
    inline reference operator()(const vector_type& offset) const
    {
      vector_type pos(cur_pos_ + offset);
      if (pos.minCoeff() < 0 || (pos-sizes_).minCoeff() >= 0)
        throw std::out_of_range("Range iterator out of range!");
      return *(cur_ptr_ + jump(offset, strides_));
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
      return *(cur_ptr_ + strides_[axis_i]*step_i);
    }

    //! Special access operator (mostly for the Hessian matrix).
    inline reference delta(int axis_i, int step_i, int axis_j, int step_j) const
    {
      return *(cur_ptr_ + strides_[axis_i]*step_i + strides_[axis_j]*step_j);
    }

    //! Special access operator (mostly for the Hessian matrix).
    template<int I, int J>
    inline reference delta(int step_i, int step_j) const
    {
      static_assert(
        I >= 0 && I < N && J >= 0 && J < N,
        "I and J must between 0 and N");
      return *(cur_ptr_ + strides_[I]*step_i + strides_[J]*step_j);
    }

    //! Axis iterator getter.
    //! The axes matches with the Cartesian view if the data is stored in a
    //! row major fashion.
    template <int Axis>
    inline AxisIterator<IsConst, T, Axis, N> axis()
    {
      return AxisIterator<IsConst, T, Axis, N>(
        cur_ptr_, cur_pos_, strides_, sizes_);
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
      return cur_pos_;
    }

    //! Return the array strides.
    inline const vector_type& strides() const
    {
      return strides_;
    }

    //! Return the array sizes.
    inline const vector_type& sizes() const
    {
      return sizes_;
    }

    //! Check if the iteration process is finished.
    inline bool end() const
    {
      return stop_;
    }

  protected: /* data members */
    bool stop_;
    pointer cur_ptr_;             //!< current pointer.
    vector_type cur_pos_;         //!< current coordinates.
    const vector_type& strides_;  //!< strides.
    const vector_type& sizes_;    //!< sizes.
  };


  //! \brief Multidimensional iterator class.
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

    using base_type::cur_pos_;
    using base_type::cur_ptr_;
    using base_type::stop_;
    using base_type::sizes_;
    using base_type::strides_;

  public:
    TYPEDEF_ITERATOR_TYPES(base_type);
    using vector_type = Matrix<int, N, 1>;
    using pointer_reference = pointer&;
    using vector_reference = vector_type&;

  public:
    inline ArrayIterator(bool stop,
                         const pointer ptr,
                         const vector_type& pos,
                         const vector_type& sizes,
                         const vector_type& strides)
      : base_type{ stop, ptr, pos, strides, sizes }
    {
    }

  public: /* iteration methods. */
    //! Prefix increment operator.
    inline self_type& operator++()
    {
      ++cur_ptr_;
      incrementer::apply(cur_pos_, stop_, sizes_);
      return *this;
    }

    //! Prefix decrement operator.
    inline self_type& operator--()
    {
      --cur_ptr_;
      decrementer::apply(cur_pos_, stop_, sizes_);
      return *this;
    }

    //! Postfix increment operator.
    inline self_type operator++(int)
    {
      self_type old{ *this };
      operator++();
      return old;
    }

    //! Postfix decrement operator.
    inline self_type operator--(int)
    {
      self_type old{ *this };
      operator--();
      return old;
    }

    //! Addition operator (slow).
    inline void operator+=(const vector_type& offset)
    {
      vector_type pos(cur_pos_ + offset);
      if (pos.minCoeff() < 0 || (pos-sizes_).minCoeff() >= 0)
        throw std::out_of_range("Range iterator out of range!");
      cur_ptr_ += jump(offset, strides_);
      cur_pos_ = pos;
    }

    //! Subtraction operator (slow).
    inline void operator-=(const vector_type& offset)
    {
      operator+=(-offset);
    }
  };


  //! \brief Multidimensional iterator base class.
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

    using base_type::stop_;
    using base_type::cur_pos_;
    using base_type::cur_ptr_;
    using base_type::strides_;

  public: /* typedefs. */
    TYPEDEF_ITERATOR_TYPES(base_type);
    typedef Matrix<int, N, 1> coords_type, vector_type;

  public: /* constructors */
    //! Constructor
    inline SubarrayIterator(bool stop, pointer ptr,
                            const vector_type& begin_pos,
                            const vector_type& end_pos,
                            const vector_type& strides,
                            const vector_type& sizes)
      : base_type{ stop, ptr+jump(begin_pos, strides), begin_pos, strides, sizes }
      , begin_{ ptr }
      , begin_pos_{ begin_pos }
      , end_pos_{ end_pos }
    {
    }

  public: /* iterator functionalities. */
    //! Prefix increment operator.
    inline self_type& operator++()
    {
      incrementer::apply(cur_pos_, stop_, begin_pos_, end_pos_);
      cur_ptr_ = begin_ + jump(cur_pos_, strides_);
      return *this;
    }

    //! Prefix decrement operator.
    inline self_type& operator--()
    {
      decrementer::apply(cur_pos_, stop_, begin_pos_, end_pos_);
      cur_ptr_ = begin_ + jump(cur_pos_, strides_);
      return *this;
    }

    //! Postfix increment operator.
    inline self_type operator++(int)
    {
      self_type old{ *this };
      operator++();
      return old;
    }

    //! Postfix increment operator.
    inline self_type operator--(int)
    {
      self_type old{ *this };
      operator--();
      return old;
    }

    //! Arithmetic operator (slow).
    inline void operator+=(const vector_type& offset)
    {
      vector_type pos{ cur_pos_ + offset };
      if ((pos-begin_pos_).minCoeff() < 0 || (pos-end_pos_).minCoeff() >= 0)
      {
        std::ostringstream msg;
        msg << "Subrange iterator out of range: pos = " << offset.transpose();
        throw std::out_of_range(msg.str().c_str());
      }
      cur_pos_ = pos;
      cur_ptr_ += jump(offset, strides_);
    }

    //! Arithmetic operator (slow).
    inline void operator-=(const vector_type& offset)
    {
      operator+=(-offset);
    }

  protected: /* data members. */
    pointer begin_;
    vector_type begin_pos_;
    vector_type end_pos_;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_ARRAYITERATORS_MULTIARRAYITERATORS_HPP */
