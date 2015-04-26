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

#ifndef DO_CORE_MULTIARRAY_MULTIARRAY_HPP
#define DO_CORE_MULTIARRAY_MULTIARRAY_HPP

#include <iostream>
#include <numeric>
#include <stdexcept>

#include <DO/Core/MultiArray/MultiArrayView.hpp>


namespace DO {

  //! The ND-array class.
  template <typename T, int N, int StorageOrder = ColMajor>
  class MultiArray : public MultiArrayView<T, N, StorageOrder>
  {
    //! @{
    //! Convenience typedefs.
    using self_type =  MultiArray;
    using base_type = MultiArrayView<T, N, StorageOrder>;
    //! @}

    using base_type::_begin;
    using base_type::_end;
    using base_type::_sizes;
    using base_type::_strides;

  public:
    using base_type::Dimension;
    using vector_type = typename base_type::vector_type;

  public: /* interface */
    //! Default constructor that constructs an empty ND-array.
    inline MultiArray()
      : base_type()
    {
    }

    //! Constructor that takes ownership of the data.
    //! The data will be cleared upon destruction of the multiarray. So make
    //! sure that is what you want. Otherwise use MultiArrayView instead.
    inline explicit MultiArray(T *data, const vector_type& sizes)
      : base_type(data, sizes)
    {
    }


    //! @{
    //! Constructor with specified sizes.
    inline explicit MultiArray(const vector_type& sizes)
    {
      initialize(sizes);
    }

    inline MultiArray(int rows, int cols)
      : MultiArray(vector_type(rows, cols))
    {
    }

    inline MultiArray(int rows, int cols, int depth)
      : MultiArray(vector_type(rows, cols, depth))
    {
    }

    //! @}

    //! Copy constructor.
    //! Create a deep copy of the other MultiArray instance.
    inline MultiArray(const self_type& other)
    {
      initialize(other._sizes);
      std::copy(other._begin, other._end, _begin);
    }

    //! \brief Copy constructor.
    //! Recopies the data of the other MultiArray with appropriate type casting.
    template <typename T2>
    inline MultiArray(const MultiArray<T2, N, StorageOrder>& other)
    {
      initialize(other.sizes());
      std::transform(other.begin(), other.end(), _begin, Cast());
    }

    //! Destructor.
    inline ~MultiArray()
    {
      delete [] _begin;
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
      // Free memory.
      if (_sizes != other.sizes())
      {
        delete[] _begin;
        // Copy everything.
        initialize(other.sizes());
      }
      std::transform(other.begin(), other.end(), _begin, Cast());
      return *this;
    }

    //! @{
    //! Resize the multi-array with the specified sizes.
    inline void resize(const vector_type& sizes)
    {
      if (_sizes != sizes)
      {
        delete[] _begin;
        initialize(sizes);
      }
    }

    inline void resize(int rows, int cols)
    {
      DO_STATIC_ASSERT(N == 2, MULTIARRAY_MUST_BE_TWO_DIMENSIONAL);
      resize(vector_type(rows, cols));
    }

    inline void resize(int rows, int cols, int depth)
    {
      DO_STATIC_ASSERT(N == 3, MULTIARRAY_MUST_BE_THREE_DIMENSIONAL);
      resize(vector_type(rows, cols, depth));
    }
    //! @}

    //! Swap multi-array objects.
    self_type& swap(self_type& other)
    {
      using std::swap;
      swap(_begin, other._begin);
      swap(_end, other._end);
      swap(_sizes, other._sizes);
      swap(_strides, other._strides);
      return *this;
    }

  private: /* helper functions for offset computation. */
    //! Construction routine.
    inline void initialize(const vector_type& sizes)
    {
      _sizes = sizes;
      bool empty = (sizes == vector_type::Zero());
      _strides = empty ? sizes : this->compute_strides(sizes);

      size_t raw_size = this->compute_size(sizes);
      _begin = empty ? 0 : new T[raw_size];
      _end = empty ? 0 : _begin + raw_size;
    }

    //! \brief Casting functor
    struct Cast
    {
      //! Overloaded operator to get the job done.
      template <typename U>
      inline T operator()(const U& u) const
      {
        return static_cast<T>(u);
      }
    };
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


#endif /* DO_CORE_MULTIARRAY_MULTIARRAY_HPP */
