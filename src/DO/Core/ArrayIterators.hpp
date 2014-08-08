// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
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

#ifndef DO_CORE_ARRAYITERATORS_HPP
#define DO_CORE_ARRAYITERATORS_HPP

#include "StaticAssert.hpp"
#include "Meta.hpp"
#include "EigenExtension.hpp"
#include <iterator>
#include <sstream>
#include <stdexcept>


namespace DO {

  //! \ingroup Core
  //! \defgroup Locator Locator
  //! @{

  //! \brief The offset computer class for N-dimensional arrays.
  template <typename Index, int N>
  inline Index jump(const Matrix<Index, N, 1>& offset,
                    const Matrix<Index, N, 1>& strides)
  {
    DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
                     INDEX_MUST_BE_INTEGRAL);
    return offset.dot(strides);
  }

  //! \brief Computes the strides from the sizes of an N-dimensional array.
  //! StorageOrder must be either 'Eigen::RowMajor' or 'Eigen::ColMajor'.
  template <int StorageOrder> struct StrideComputer;
  //! \brief Increment the current position in an N-dimensional array.
  //! StorageOrder must be either 'Eigen::RowMajor' or 'Eigen::ColMajor'.
  template <int StorageOrder> struct PositionIncrementer;
  //! \brief Decrement the current position in an N-dimensional array.
  //! StorageOrder must be either 'Eigen::RowMajor' or 'Eigen::ColMajor'.
  template <int StorageOrder> struct PositionDecrementer;

  template <> struct StrideComputer<RowMajor>
  {
    template <typename Index, int N>
    static Matrix<Index, N, 1> eval(const Matrix<Index, N, 1>& sizes)
    {
      DO_STATIC_ASSERT(N > 0, N_MUST_BE_POSITIVE);
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
                       INDEX_MUST_BE_INTEGRAL);
      Matrix<Index, N, 1> strides;
      strides[N-1] = 1;
      for (int i = N-2; i >= 0; --i)
        strides[i] = strides[i+1]*sizes[i+1];
      return strides;
    }
  };
  template <> struct StrideComputer<ColMajor>
  {
    template <typename Index, int N>
    static Matrix<Index, N, 1> eval(const Matrix<Index, N, 1>& sizes)
    {
      DO_STATIC_ASSERT(N > 0, N_MUST_BE_POSITIVE);
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
                       INDEX_MUST_BE_INTEGRAL);
      Matrix<Index, N, 1> strides;
      strides[0] = 1;
      for (int i = 1; i < N; ++i)
        strides[i] = strides[i-1]*sizes[i-1];
      return strides;
    }
  };

  template <> struct PositionIncrementer<RowMajor>
  {
    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end)
    {
      for (int i = N-1; i >= 0; --i)
      {
        ++coords[i];
        if (coords[i] != end[i])
          return;
        coords[i] = start[i];
      }
      if (coords[0] == start[0])
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& sizes)
    { apply<Index, N>(coords, stop, Matrix<Index, N, 1>::Zero(), sizes); }

  };
  template <> struct PositionIncrementer<ColMajor>
  {
    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end)
    {
      for (int i = 0; i < N; ++i)
      {
        ++coords[i];
        if (coords[i] != end[i])
          return;
        coords[i] = start[i];
      }
      if (coords[N-1] == start[N-1])
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& sizes)
    { apply<Index, N>(coords, stop, Matrix<Index, N, 1>::Zero(), sizes); }
  };

  template <> struct PositionDecrementer<RowMajor>
  {
    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end)
    {
      for (int i = N-1; i >= 0; --i)
      {
        if (coords[i] != start[i])
        {
          --coords[i];
          return;
        }
        coords[i] = end[i]-1;
      }
      if (coords[0] == end[0]-1)
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& sizes)
    { apply<Index, N>(coords, stop, Matrix<Index, N, 1>::Zero(), sizes); }
  };
  template <> struct PositionDecrementer<ColMajor>
  {
    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end)
    {
      for (int i = 0; i < N; ++i)
      {
        if (coords[i] != start[i])
        {
          --coords[i];
          return;
        }
        coords[i] = end[i]-1;
      }
      if (coords[N-1] == end[N-1]-1)
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& sizes)
    { apply<Index, N>(coords, stop, Matrix<Index, N, 1>::Zero(), sizes); }
  };

  // ======================================================================== //
  // Forward declaration of multidimensional iterators.
  template <bool IsConst, typename T, int N, int StorageOrder>
  class ArrayIteratorBase;
  template <bool IsConst, typename T, int N, int StorageOrder>
  class ArrayIterator;
  template <bool IsConst, typename T, int N, int StorageOrder>
  class SubarrayIterator;
  template <bool IsConst, typename T, int Axis, int N>
  class AxisIterator;

  // ======================================================================== //
  // Convenient typedefs.
#define ITERATOR_BASE_TYPE(IsConst)                       \
  std::iterator<                                          \
    std::random_access_iterator_tag, T, std::ptrdiff_t,   \
    typename Meta::Choose<IsConst, const T *, T *>::Type, \
    typename Meta::Choose<IsConst, const T&, T&>::Type    \
  >

#define TYPEDEF_ITERATOR_TYPES(IteratorType)                      \
  typedef typename base_type::value_type value_type;              \
  typedef typename base_type::difference_type difference_type;    \
  typedef typename base_type::pointer pointer;                    \
  typedef typename base_type::reference reference;                \
  typedef typename base_type::iterator_category iterator_category


  // ======================================================================== //
  //! \brief Axis iterator class for N-dimensional arrays.
  template <bool IsConst, typename T, int Axis, int N>
  class AxisIterator : public ITERATOR_BASE_TYPE(IsConst)
  {
    DO_STATIC_ASSERT(
      Axis >= 0 && Axis < N,
      AXIS_MUST_BE_NONNEGATIVE_AND_LESS_THAN_N);

    // Friend classes.
    template <bool, typename, int, int> friend class AxisIterator;
    template <bool, typename, int, int> friend class ArrayIteratorBase;

    // Private typedefs.
    typedef ITERATOR_BASE_TYPE(IsConst) base_type;
    typedef AxisIterator self_type;

  public: /* STL-like typedefs */
    TYPEDEF_ITERATOR_TYPES(base_type);
    typedef Matrix<int, N, 1> vector_type;

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
    {}

  public: /* dereferencing, access functions. */
    //! Dereferencing operator.
    inline reference operator*() const
    { return *cur_ptr_; }
    //! Dereferencing operator.
    inline pointer operator->() const
    { return cur_ptr_; }
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
    { return cur_ptr_ == rhs.cur_ptr_; }
    template <bool IsConst2, int StorageOrder>
    inline bool operator==(const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& rhs) const
    { return cur_ptr_ == rhs.cur_ptr_; }
    //! Equality operator.
    inline bool operator==(const T *ptr) const
    { return cur_ptr_ == ptr; }
    //! Inequality operator.
    template <bool IsConst2, int Axis2>
    inline bool operator!=(const AxisIterator<IsConst2, T, Axis2, N>& other) const
    { return !operator==(other); }
    template <bool IsConst2, int StorageOrder>
    inline bool operator!=(const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& other) const
    { return !operator==(other); }
    //! Inequality operator.
    inline bool operator!=(const T *ptr) const
    { return !operator==(ptr); }

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
    { return operator+=(-n); }

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
      AxisIterator old(*this);
      operator++();
      return old;
    }
    //! Postfix decrement operator.
    inline self_type operator--(int)
    {
      AxisIterator old(*this);
      operator--();
      return old;
    }

  private: /* data members. */
    pointer& cur_ptr_;            //!< current pointer.
    vector_type& cur_pos_;        //!< current coordinates.
    const vector_type& strides_;  //!< strides.
    const vector_type& sizes_;    //!< sizes.
  };


  // ======================================================================== //
  //! \brief Multidimensional iterator base class.
  //! The 'ArrayIteratorBase' class is a heavy object. It is mostly useful
  //! for differential calculus. When possible, prefer using more elementary
  //! iterator instead.
  template <bool IsConst, typename T, int N, int StorageOrder>
  class ArrayIteratorBase : public ITERATOR_BASE_TYPE(IsConst)
  {
    typedef ITERATOR_BASE_TYPE(IsConst) base_type;
    template <bool, typename, int, int> friend class AxisIterator;
    template <bool, typename, int, int> friend class ArrayIteratorBase;

  public: /* typedefs */
    TYPEDEF_ITERATOR_TYPES(base_type);
    typedef ArrayIteratorBase self_type;
    typedef Matrix<int, N, 1> vector_type;
    typedef AxisIterator<IsConst, value_type, 0, N> x_iterator;
    typedef AxisIterator<IsConst, value_type, 1, N> y_iterator;
    typedef AxisIterator<IsConst, value_type, 2, N> z_iterator;

  public: /* interface */
    //! Constructor
    inline ArrayIteratorBase(bool stop,
                             pointer ptr,
                             const vector_type& pos,
                             const vector_type& strides,
                             const vector_type& sizes)
      : base_type()
      , stop_(stop)
      , cur_ptr_(ptr)
      , cur_pos_(pos)
      , strides_(strides)
      , sizes_(sizes)
    {
    }

  public: /* dereferencing functions. */
    //! Dereferencing operator.
    inline reference operator*() const
    { return *cur_ptr_; }
    //! Referencing operator.
    inline pointer operator->() const
    { return cur_ptr_; }

    public: /* equality/inequality comparisons. */
    //! Equality operator.
    inline bool operator==(const self_type& other) const
    { 
      if (stop_ && other.stop_ )
        return true;
      if (cur_ptr_ == other.cur_ptr_)
        return true;
      return false;
    }
    //! Equality operator.
    template <bool IsConst2>
    inline bool operator==(
      const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& other
    ) const
    { 
      if (stop_ && other.stop_)
        return true;
      if (cur_ptr_ == other.cur_ptr_)
        return true;
      return false;
    }
    //! Equality operator.
    inline bool operator==(const T *ptr) const
    { return cur_ptr_ == ptr; }
    //! Inequality operator.
    inline bool operator!=(const self_type& other) const
    { return !this->operator==(other); }
    //! Inequality operator.
    template <bool IsConst2>
    inline bool operator!=(
      const ArrayIteratorBase<IsConst2, T, N, StorageOrder>& other
    ) const
    { return !this->operator==(other); }
    //! Inequality operator.
    inline bool operator!=(const T *ptr) const
    { return !operator==(ptr); }

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
      DO_STATIC_ASSERT(N == 2, DATA_MUST_BE_TWO_DIMENSIONAL);
      return operator()(vector_type(i, j));
    }
    //! Special access operator in 3D.
    inline reference operator()(int i, int j, int k) const
    {
      DO_STATIC_ASSERT(N == 3, DATA_MUST_BE_THREE_DIMENSIONAL);
      return operator()(vector_type(i, j, k));
    }
    //! Special access operator (mostly for the hessian matrix).
    inline reference delta(int i, int di, int j, int dj) const
    { return *(cur_ptr_ + strides_[i]*di + strides_[j]*dj); }
    //! Special access operator (mostly for the hessian matrix).
    template<int I, int J>
    inline reference delta(int di, int dj) const
    {
      DO_STATIC_ASSERT(I >= 0 && I < N && J >= 0 && J < N,
                       I_AND_J_MUST_BETWEEN_0_AND_N);
      return *(cur_ptr_ + strides_[I]*di + strides_[J]*dj);
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
    inline x_iterator x() { return axis<0>(); }
    //! Y-axis iterator getter.
    inline y_iterator y() { return axis<1>(); }
    //! Z-Axis iterator getter.
    inline z_iterator z() { return axis<2>(); }

  public: /* additional features. */
    //! Return the current position in the the array.
    inline const vector_type& position() const
    { return cur_pos_; }
    //! Return the array strides.
    inline const vector_type& strides() const
    { return strides_; }
    //! Return the array sizes.
    inline const vector_type& sizes() const
    { return sizes_; }

  protected: /* data members */
    bool stop_;
    pointer cur_ptr_;             //!< current pointer.
    vector_type cur_pos_;         //!< current coordinates.
    const vector_type& strides_;  //!< strides.
    const vector_type& sizes_;    //!< sizes.
  };


  // ======================================================================== //
  //! \brief Multidimensional iterator class.
  //! The 'ArrayIterator' class is a heavy object. It is mostly useful for
  //! differential calculus. When possible, prefer using more elementary
  //! iterator instead.
  template <bool IsConst, typename T, int N, int StorageOrder>
  class ArrayIterator : public ArrayIteratorBase<IsConst, T, N, StorageOrder>
  {
    DO_STATIC_ASSERT(N >= 0, NUMBER_OF_DIMENSIONS_MUST_BE_NONNEGATIVE);
    typedef ArrayIteratorBase<IsConst, T, N, StorageOrder> base_type;
    typedef ArrayIterator self_type;
    typedef PositionIncrementer<StorageOrder> incrementer;
    typedef PositionDecrementer<StorageOrder> decrementer;
    
    using base_type::cur_pos_;
    using base_type::cur_ptr_;
    using base_type::stop_;
    using base_type::sizes_;
    using base_type::strides_;

  public:
    TYPEDEF_ITERATOR_TYPES(base_type);
    typedef Matrix<int, N, 1> vector_type;
    typedef pointer& pointer_reference;
    typedef vector_type& vector_reference;

  public:
    inline ArrayIterator(bool stop,
                         const pointer ptr,
                         const vector_type& pos,
                         const vector_type& sizes,
                         const vector_type& strides)
      : base_type(stop, ptr, pos, strides, sizes) {}

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
      self_type old(*this);
      operator++();
      return old;
    }
    //! Postfix decrement operator.
    inline self_type operator--(int)
    {
      self_type old(*this);
      operator--();
      return old;
    }

  public: /* random jump methods. */
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
    { operator+=(-offset); }
  };


  // ======================================================================== //
  //! \brief Multidimensional iterator base class.
  //! The 'SubarrayIterator' class is a heavy object. It is mostly useful for
  //! differential calculus. When possible, prefer using more elementary
  //! iterator instead.
  template <bool IsConst, typename T, int N, int StorageOrder = ColMajor>
  class SubarrayIterator : public ArrayIteratorBase<IsConst, T, N, StorageOrder>
  {
    DO_STATIC_ASSERT(N >= 0, NUMBER_OF_DIMENSIONS_MUST_BE_NONNEGATIVE);

    typedef ArrayIteratorBase<IsConst, T, N, StorageOrder> base_type;
    typedef SubarrayIterator self_type;
    typedef PositionIncrementer<StorageOrder> incrementer;
    typedef PositionDecrementer<StorageOrder> decrementer;
    
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
      : base_type(stop, ptr+jump(begin_pos, strides), begin_pos, strides, sizes)
      , begin_(ptr)
      , begin_pos_(begin_pos)
      , end_pos_(end_pos)
    {}

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
      self_type old(*this);
      operator++();
      return old;
    }
    //! Postfix increment operator.
    inline self_type operator--(int)
    {
      self_type old(*this);
      operator--();
      return old;
    }
    //! Arithmetic operator (slow).
    inline void operator+=(const vector_type& offset)
    {
      vector_type pos(cur_pos_ + offset);
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
    { operator+=(-offset); }

  protected: /* data members. */
    pointer begin_;
    vector_type begin_pos_;
    vector_type end_pos_;
  };

  //! @}

} /* namespace DO */

#endif /* DO_CORE_ARRAYITERATORS_HPP */