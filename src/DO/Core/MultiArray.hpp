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
//! \brief This contains the implementation of the N-dimensional array class.

#ifndef DO_CORE_MULTIARRAY_HPP
#define DO_CORE_MULTIARRAY_HPP

#include "Locator.hpp"
#include <iostream>
#include <numeric>

namespace DO {

  //! \ingroup Core
  //! \defgroup MultiArray MultiArray
  //! @{

  //! The generic traits class for the MultiArray element type.
  //! This traits class is when the array/matrix view is used. This serves as
  //! an interface for the Eigen library.
  template <typename T>
  struct ElementTraits
  {
    typedef T value_type; //!< STL-like typedef.
    typedef size_t size_type; //!< STL-like typedef.
    typedef T * pointer; //!< STL-like typedef.
    typedef const T * const_pointer; //!< STL-like typedef.
    typedef T& reference; //!< STL-like typedef.
    typedef const T& const_reference; //!< STL-like typedef.
    typedef T * iterator; //!< STL-like typedef.
    typedef const T * const_iterator; //!< STL-like typedef.
    static const bool is_scalar = true; //!< STL-like typedef.
  };

  //! \brief The specialized element traits class when the entry is a non square
  //! matrix. Again the matrix is viewed as a 'super-scalar'.
  //! However, super-scalar operations will be regular point-wise matrix
  //! operations.
  //! Therefore this applies for row-vector and column vector.
  template <typename T, int M, int N>
  struct ElementTraits<Matrix<T, M, N> >
  {
    const static bool is_square_matrix = (M == N); //!< STL-like typedef.
    typedef typename Meta::Choose<
      is_square_matrix,
      Matrix<T, N, N>,
      Array<T, M, N> >::Type value_type; //!< STL-like typedef.
    typedef size_t size_type; //!< STL-like typedef.
    typedef value_type * pointer; //!< STL-like typedef.
    typedef const value_type * const_pointer; //!< STL-like typedef.
    typedef value_type& reference; //!< STL-like typedef.
    typedef const value_type& const_reference; //!< STL-like typedef.
    typedef value_type * iterator; //!< STL-like typedef.
    typedef const value_type * const_iterator; //!< STL-like typedef.
    static const bool is_scalar = false; //!< STL-like typedef.
  };

  //! \brief The specialized element traits class when the entry is an array.
  //! Default super-scalar operations are point-wise matrix operations.
  template <typename T, int M, int N>
  struct ElementTraits<Array<T, M, N> >
  {
    typedef Array<T, M, N> value_type; //!< STL-like typedef.
    typedef size_t size_type; //!< STL-like typedef.
    typedef value_type * pointer; //!< STL-like typedef.
    typedef const value_type * const_pointer; //!< STL-like typedef.
    typedef value_type& reference; //!< STL-like typedef.
    typedef const value_type& const_reference; //!< STL-like typedef.
    typedef value_type * iterator; //!< STL-like typedef.
    typedef const value_type * const_iterator; //!< STL-like typedef.
    static const bool is_scalar = false; //!< STL-like typedef.
  };

  /*!
    \brief The N-dimensional array class.

    \todo:
    - prefer shallow copy in copy constructor and in the assignment operator
    - add 'MultiArray MultiArray::clone() const' method.
      (Performance still seems acceptable even if deep copies are always done.)
    - **DOUBLE-CHECK** all existing image-based algorithm.
   */
  template <typename T, int N, int StorageOrder_ = ColMajor>
  class MultiArray
  {
  private: /* STL-like interface. */
    typedef MultiArray self_type;

  public: /* STL-like interface. */
    enum { StorageOrder = StorageOrder_ };
    typedef std::size_t size_type;          //!< STL typedef.
    typedef std::ptrdiff_t difference_type; //!< STL typedef.
    typedef T value_type;                   //!< STL typedef.
    typedef T * pointer;                    //!< STL typedef.
    typedef const T * const_pointer;        //!< STL typedef.
    typedef T& reference;                   //!< STL typedef.
    typedef const T& const_reference;       //!< STL typedef.
    typedef T * iterator;                   //!< STL typedef.
    typedef const T * const_iterator;       //!< STL typedef.

    //! Vector type.
    typedef Matrix<int, N, 1> vector_type;

    //! N-dimensional iterator.
    typedef RangeIterator<false, T, N, StorageOrder> range_iterator;
    //! N-dimensional subrange iterator.
    typedef SubrangeIterator<false, T, N, StorageOrder> subrange_iterator;
     //! Immutable N-dimensional iterator.
    typedef RangeIterator<true, T, N, StorageOrder> const_range_iterator;
    //! Immutable N-dimensional subrange iterator.
    typedef SubrangeIterator<true, T, N, StorageOrder> const_subrange_iterator;

    //! Immutable matrix view for linear algebra.
    typedef Map<const Array<typename ElementTraits<T>::value_type, Dynamic, 1> >
      const_array_view_type;
    //! Mutable matrix view for linear algebra.
    typedef Map<Array<typename ElementTraits<T>::value_type, Dynamic, 1> >
      array_view_type;
    //! Immutable matrix view for linear algebra.
    typedef Map<const Matrix<typename ElementTraits<T>::value_type,
      Dynamic, Dynamic, StorageOrder> > const_matrix_view_type;
    //! Mutable matrix view for linear algebra.
    typedef Map<Matrix<typename ElementTraits<T>::value_type,
      Dynamic, Dynamic, StorageOrder> > matrix_view_type;

  public: /* interface */
    //! Default constructor that constructs an empty N-dimensional array.
    inline MultiArray()
    { initialize(vector_type::Zero()); }
    //! Constructor that wraps plain data with its known sizes.
    inline MultiArray(value_type *data, const vector_type& sizes,
                      bool get_data_ownership = false)
      : begin_(data)
      , end_(data+compute_size(sizes))
      , sizes_(sizes), strides_(compute_strides(sizes))
      , has_data_ownership_(get_data_ownership)
    {}
    //! \brief Default constructor that allocates an N-dimensional array with
    //! the specified sizes.
    inline explicit MultiArray(const vector_type& sizes)
    { initialize(sizes); }
    //! \brief Default constructor that allocates a 2D array with
    //! the specified rows and columns.
    inline MultiArray(int rows, int cols)
    { 
      DO_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_BE_TWO_DIMENSIONAL);
      initialize(Vector2i(rows, cols));
    }
    //! \brief Default constructor that allocates a 3D array with
    //! the specified rows, columns and depth.
    inline MultiArray(int rows, int cols, int depth)
    {
      DO_STATIC_ASSERT(N == 3, MULTIARRAY_MUST_BE_THREE_DIMENSIONAL);
      initialize(Vector3i(rows, cols, depth));
    }
    //! Copy constructor that makes a deep copy of the source array.
    inline MultiArray(const self_type& other)
    {
      initialize(other.sizes_);
      std::copy(other.begin_, other.end_, begin_);
    }
    //! \brief Copy constructor that recopies the data source array with
    //! appropriate type casting.
    template <typename T2>
    inline MultiArray(const MultiArray<T2, N, StorageOrder>& other)
    {
      initialize(other.sizes());
      std::transform(other.begin(), other.end(), begin_, Cast());
    }

    //! Destructor.
    inline ~MultiArray()
    {
      if (has_data_ownership_ && begin_)
        delete [] begin_;
    }

    //! \brief Assignment operator uses the copy-swap idiom.
    self_type& operator=(self_type other)
    {
      swap(other);
      return *this;
    }
    //! \brief Assignment operator that recopies the content of the source array
    //! with appropriate type casting.
    template <typename T2>
    const self_type& operator=(const MultiArray<T2, N, StorageOrder>& other)
    {
      if (!has_data_ownership_)
      {
        std::cerr << "Fatal Error: using the assignment operator on wrapped data is not allowed!" << std::endl;
        std::cerr << "Terminating !" << std::endl;
        exit(-1);
      }
      // Free memory.
      if (begin_)
        delete[] begin_;
      // Copy everything.
      initialize(other.sizes());
      std::transform(other.begin(), other.end(), begin_, Cast());
      return *this;
    }

    //! Mutable referencing operator.
    inline reference operator()(const vector_type& pos)
    { return begin_[offset(pos)]; }
    //! Mutable referencing operator.
    inline reference operator()(int i, int j)
    {
      DO_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_BE_TWO_DIMENSIONAL);
      return begin_[offset(Vector2i(i, j))];
    }
    //! Mutable referencing operator.
    inline reference operator()(int i, int j, int k)
    {
      DO_STATIC_ASSERT(N == 3, MULTIARRAY_MUST_BE_THREE_DIMENSIONAL);
      return begin_[offset(Vector3i(i, j, k))];
    }
    //! Non-mutable referencing operator.
    inline const_reference operator()(const vector_type& pos) const
    { return begin_[offset(pos)]; }
    //! Non-mutable referencing operator.
    inline const_reference operator()(int i, int j) const
    {
      DO_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_BE_TWO_DIMENSIONAL);
      return begin_[offset(Vector2i(i, j))];
    }
    //! Non-mutable referencing operator.
    inline const_reference operator()(int i, int j, int k) const
    {
      DO_STATIC_ASSERT(N == 3, MULTIARRAY_MUST_BE_THREE_DIMENSIONAL);
      return begin_[offset(Vector3i(i, j, k))];
    }

    //! Mutable POD accessor.
    inline pointer data()
    { return begin_; }
    //! Non-mutable POD accessor.
    inline const_pointer data() const
    { return begin_; }

    //! Mutable beginning iterator.
    inline iterator begin()
    { return begin_; }
    //! Non-mutable beginning iterator.
    inline const_iterator begin() const
    { return begin_; }
    //! Mutable ending iterator.
    inline iterator end()
    { return end_; }
    //! Non-mutable ending iterator.
    inline const_iterator end() const
    { return end_; }

    //! Size getter.
    const vector_type& sizes() const
    { return sizes_; }
    //! Raw size getter.
    size_type size() const
    { return end_ - begin_; }
    //! Size getter along the i-th.
    int size(int i) const
    { return sizes_[i]; }
    //! Number of rows getter.
    int rows() const
    { return sizes_[0]; }
    //! Number of cols getter.
    int cols() const
    { return sizes_[1]; }
    //! Depth getter.
    int depth() const
    { return sizes_[2]; }

    //! Strides getter.
    inline const vector_type& strides() const
    { return strides_; }
    //! Stride along the i-th dimension.
    inline int stride(int i) const
    { return strides_[i]; }

    //! Mutable begin range iterator.
    inline range_iterator begin_range()
    { return range_iterator(false, begin_, vector_type::Zero(), sizes_, strides_); }
    //! Mutable end range iterator.
    inline range_iterator end_range()
    { return range_iterator(true, end_, sizes_, sizes_, strides_); }
    //! Mutable begin subrange iterator.
    inline subrange_iterator begin_subrange(const vector_type& start,
                                            const vector_type& end)
    { return subrange_iterator(false, begin_, start, end, strides_, sizes_); }
    //! Mutable begin subrange iterator.
    inline subrange_iterator end_subrange()
    {
      return subrange_iterator(
        true, begin_, sizes_, sizes_, strides_, sizes_);
    }
    //! Immutable begin range iterator.
    inline const_range_iterator begin_range() const
    {
      return const_range_iterator(
        false, begin_, vector_type::Zero(), sizes_, strides_);
    }
    //! Immutable end range iterator.
    inline const_range_iterator end_range() const
    { return const_range_iterator(true, end_, sizes_, sizes_, strides_); }
    //! Immutable begin subrange iterator.
    inline const_subrange_iterator begin_subrange(const vector_type& start,
                                                  const vector_type& end) const
    {
      return const_subrange_iterator(
        false, begin_, start, end, strides_, sizes_);
    }
    //! Immutable end subrange iterator.
    inline const_subrange_iterator end_subrange() const
    {
        return const_subrange_iterator(
          true, begin_, sizes_, sizes_, strides_, sizes_);
    }

    //! Resizing method.
    inline bool resize(const vector_type& sizes)
    {
      if (!has_data_ownership_)
      {
        std::cerr << "data is wrapped! Not resizing" << std::endl;
        return false;
      }

      if (!begin_)
        delete[] begin_;
      initialize(sizes);
      return true;
    }
    //! Resizing method.
    inline bool resize(int rows, int cols)
    {
      DO_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_BE_TWO_DIMENSIONAL);
      return resize(vector_type(rows, cols));
    }
    //! Resizing method.
    inline bool resize(int rows, int cols, int depth)
    {
      DO_STATIC_ASSERT(N == 3, MULTIARRAY_MUST_BE_THREE_DIMENSIONAL);
      return resize(vector_type(rows, cols, depth));
    }

    //! Non-mutable array view for linear algebra with Eigen 3.
    inline const_array_view_type array() const
    {
      return const_array_view_type( reinterpret_cast<
        const typename ElementTraits<T>::const_pointer>(data()),
        size());
    }
    //! Mutable array view for linear algebra with Eigen 3.
    inline array_view_type array()
    {
      return array_view_type( reinterpret_cast<
        typename ElementTraits<T>::pointer>(data()),
        size());
    }
    //! Non-mutable matrix view for linear algebra with Eigen 3.
    inline const_matrix_view_type matrix() const
    {
      DO_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_HAVE_TWO_DIMENSIONS);
      return const_matrix_view_type( reinterpret_cast<
        typename ElementTraits<T>::const_pointer>(data()),
        rows(), cols());
    }
    //! Mutable matrix view for linear algebra with Eigen 3.
    inline matrix_view_type matrix()
    {
      DO_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_HAVE_TWO_DIMENSIONS);
      return matrix_view_type( reinterpret_cast<
        typename ElementTraits<T>::pointer>(data()),
        rows(), cols());
    }

    //! Swap arrays.
    self_type& swap(self_type& other)
    {
      using std::swap;
      swap(begin_, other.begin_);
      swap(end_, other.end_);
      swap(sizes_, other.sizes_);
      swap(strides_, other.strides_);
      swap(has_data_ownership_, other.has_data_ownership_);
      return *this;
    }

  private: /* helper functions for offset computation. */
    //! \brief Stride computing method called in the construction and
    //! resizing of the array.
    inline vector_type compute_strides(const vector_type& sizes)
    { return StrideComputer<StorageOrder>::eval(sizes); }
    //! \brief Raw size computing method called in the construction and
    //! resizing of the array.
    inline int compute_size(const vector_type& sizes) const
    {
      return std::accumulate(sizes.data(), sizes.data()+N,
        1, std::multiplies<int>());
    }
    //! Offset computing method.
    inline int offset(const vector_type& pos) const
    { return jump(pos, strides_); }
    //! Construction routine.
    inline void initialize(const vector_type& sizes)
    {
      sizes_ = sizes;
      bool empty = (sizes == vector_type::Zero());
      strides_ = empty ? sizes : compute_strides(sizes);

      has_data_ownership_ = true;
      size_t raw_size = compute_size(sizes);
      begin_ = empty ? 0 : new T[raw_size];
      end_ = empty ? 0 : begin_ + raw_size;
    }
    //! \brief Casting functor
    struct Cast
    {
      //! Overloaded operator to get the job done.
      template <typename U>
      inline T operator()(const U& u) const { return static_cast<T>(u); }
    };

  private: /* data members. */
    value_type *begin_;       //!< first element of the data.
    value_type *end_;         //!< last element of the data.
    vector_type sizes_;       //!< vector of size along each dimension.
    vector_type strides_;     //!< vector of stride for each dimension.
    //! \brief flag that checks if the array wraps some data. It is used for
    //! deallocation.
    bool has_data_ownership_;
  };

  //! output stream operator
  template <typename T, int N, int StorageOrder>
  std::ostream& operator<<(std::ostream& os,
                           const MultiArray<T,N,StorageOrder>& M)
  {
    os << M.sizes() << std::endl;
    os << M.array() << std::endl;
    return os;
  }

  //! @}

}

#endif /* DO_CORE_MULTIARRAY_HPP */