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

#ifndef DO_CORE_LOCATOR_HPP
#define DO_CORE_LOCATOR_HPP

#include "StaticAssert.hpp"
#include "Meta.hpp"
#include "EigenExtension.hpp"
#include <iterator>
#include <iostream>

namespace DO {

  //! \ingroup Core
  //! \defgroup Locator Locator
  //! @{

  //! \brief The offset computer class for N-dimensional arrays.
  //! StorageOrder must be either 'Eigen::RowMajor' or 'Eigen::ColMajor'.
  template <int N, int StorageOrder = RowMajor> struct Offset;
  template <int N> struct Offset2;

  //! \brief The specialized offset computer for dimension N > 1 and 
  //! row-major storage.
  template <int N>
  struct Offset<N, RowMajor>
  {
    //! Compile-time instantiation check.
    DO_STATIC_ASSERT(N > 0, N_MUST_BE_POSITIVE);
    //! Computes the index that corresponds to the coordinates of an ND array by
    //! loop unrolling.
    template <typename Index>
    static inline Index eval(const Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return coords[N-1] + size[N-1]*Offset<N-1, RowMajor>::eval(coords, size);
    }
    //! Computes the index that corresponds to the coordinates of a 2D array.
    template <typename Index>
    static inline Index eval(Index i, Index j, Index size1, Index size2)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return i*size2 + j;
    }
    //! Computes the index that corresponds to the coordinates of a 3D array.
    template <typename Index>
    static inline Index eval(Index i, Index j, Index k,
                             Index size1, Index size2, Index size3)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return (i*size2 + j) * size3 + k;
    }
    //! Computes the strides from the sizes of an ND array.
    //! It is not critical to unroll the underlying loop in this function as 
    //! the strides are normally computed once and for all.
    template <typename Index>
    static inline void eval_strides(Index *strides, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      strides[N-1] = 1;
      for (int i = N-2; i >= 0; --i)
        strides[i] = strides[i+1]*size[i+1];
    }
    //! Computes the coordinates from the index.
    template <typename Index>
    static inline void eval_coords_from_offset(Index *coords, Index offset,
                                               const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      /*for (int i = N-1; i > 1; --i)
      {
        coords[i] = offset % dims[i];
        offset = (offset - coords[i])/dims[i];
      }
      coords[0] = (offset - coords[1]) / dims[1];*/
      Offset<N, RowMajor>::eval_coords_from_offset_(coords+N-1, offset, dims+N-1);
    }
    //! Unrolls the underlying loop of 'eval_coords_from_offset'.
    template <typename Index>
    static inline void eval_coords_from_offset_(Index *coords, 
                                                Index offset,
                                                const Index *dims)
    {
      *coords = offset % *dims;
      offset = (offset - *coords) / *dims;
      --coords; --dims;
      Offset<N-1, RowMajor>::eval_coords_from_off_(coords, offset, dims);
    }
    //! Computes the incremented coordinates.
    template <typename Index>
    static inline void increment_coords(Index *coords, const Index *size,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      //for (int axis = n-1; axis >= 0; --axis)
      //{
      //  ++coords[axis];
      //  if (coords[axis] < size[axis])
      //    break;
      //  coords[axis] = 0;
      //}
      Offset<N, RowMajor>::increment_coords_(coords+N-1, size+N-1, stop);
    }
    //! Unrolls the underlying loop of 'increment_coords'.
    template <typename Index>
    static inline void increment_coords_(Index *coords, const Index *size,
                                         bool& stop)
    {
      // precondition:
      // *coords < *size
      ++(*coords);
      if (*coords != *size)
        return;
      *coords = 0;
      --coords; --size;
      Offset<N-1>::increment_coords_(coords, size, stop);
    }
    //! Computes the incremented coordinates.
    template <typename Index>
    static inline void increment_coords(Index *coords,
                                        const Index *start, const Index *end,
                                        bool& stop)
    {
      Offset<N, RowMajor>::increment_coords_(coords+N-1, start+N-1, end+N-1,
                                             stop);
    }
    //! Unrolls the underlying loop of 'increment_coords'.
    template <typename Index>
    static inline void increment_coords_(Index *coords,
                                         const Index *start, const Index *end,
                                         bool& stop)
    {
      // precondition:
      // *start <= *coords && *coords < *end
      ++(*coords);
      if (*coords != *end)
        return;
      *coords = *start;
      --coords; --start; --end;
      Offset<N-1, RowMajor>::increment_coords_(coords, start, end, stop);
    }
    //! Computes the decremented coordinates.
    template <typename Index>
    static inline void decrement_coords(Index *coords, const Index *size,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      //for (int axis = N-1; axis >= 0; --axis)
      //{
      //  if (coords[axis] > 0 && coords[axis] != size[axis])
      //  {
      //    --coords[axis];
      //    break;
      //  }
      //  coords[axis] = size[axis]-1;
      //}
      Offset<N, RowMajor>::decrement_coords_(coords+N-1, size+N-1, stop);
    }
    //! Unrolls the underlying loop of 'decrement_coords'.
    template <typename Index>
    static inline void decrement_coords_(Index *coords, const Index *size,
                                         bool& stop)
    {
      // precondition:
      // 0 <= *coords && *coords < *size
      if (*coords > 0)
      {
        --(*coords);
        return;
      }
      *coords = *size-1;
      --coords; --size;
      Offset<N-1>::decrement_coords_(coords, size, stop);
    }
    //! Computes the decremented coordinates.
    template <typename Index>
    static inline void decrement_coords(Index *coords,
                                        const Index *start, const Index *end,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      //for (int axis = N-1; axis >= 0; --axis)
      //{
      //  if (coords[axis] > 0 && coords[axis] != size[axis])
      //  {
      //    --coords[axis];
      //    break;
      //  }
      //  coords[axis] = size[axis]-1;
      //}
      Offset<N, RowMajor>::decrement_coords_(coords+N-1, start+N-1, end+N-1,
                                             stop);
    }
    //! Unrolls the underlying loop of 'decrement_coords'.
    template <typename Index>
    static inline void decrement_coords_(Index *coords,
                                         const Index *start, const Index *end,
                                         bool& stop)
    {
      // precondition:
      // *start <= *coords && *coords < *end
      if (*coords > *start)
      {
        --(*coords);
        return;
      }
      *coords = *end-1;
      --coords; --start; --end;
      Offset<N-1>::decrement_coords_(coords, start, end, stop);
    }
  };

  //! \brief The specialized offset computer for dimension 1 and 
  //! row-major storage (for loop unrolling).
  template <>
  struct Offset<1, RowMajor>
  {
    //! Merely returns the 1D coordinates.
    template <typename Index>
    static inline Index eval(const Index *coords, const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return *coords;
    }
    //! Merely returns 1.
    template <typename Index>
    static inline void eval_strides(Index *strides, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      *strides = 1;
    }
    //! Merely returns the offset value.
    template <typename Index>
    static inline void eval_coords_from_offset(Index *coords, Index offset,
                                               const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      *coords = offset;
    }
    //! Finishes the unrolled loop of 'eval_coords_from_offset'.
    template <typename Index>
    static inline void eval_coords_from_offset_(Index *coords, Index offset,
                                                const Index *dims)
    {
      *coords = (offset - *(coords+1)) / *(dims+1);
    }
    //! Merely does +1.
    template <typename Index>
    static inline void increment_coords(Index *coords, const Index *size,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      if (*coords == *size-1)
      {
        stop = true;
        return;
      }
      ++(*coords);
    }
    //! Finishes the underlying loop of 'increment_coords'.
    template <typename Index>
    static inline void increment_coords_(Index *coords, const Index *size,
                                         bool& stop)
    {
      if (*coords == *size-1)
      {
        stop = true;
        return;
      }
      ++(*coords);
    }
    //! Finishes the incremented coordinates.
    template <typename Index>
    static inline void increment_coords(Index *coords,
                                        const Index *start, const Index *end,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      if (*coords == *end-1)
      {
        stop = true;
        return;
      }
      ++(*coords);
    }
    //! Finishes the underlying loop of 'increment_coords'.
    template <typename Index>
    static inline void increment_coords_(Index *coords,
                                         const Index *start, const Index *end,
                                         bool& stop)
    {
      if (*coords == *end-1)
      {
        stop = true;
        return;
      }
      ++(*coords);
    }
    //! Merely does -1.
    template <typename Index>
    static inline void decrement_coords(Index *coords, const Index *size,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      if (*coords == 0)
      {
        stop = true;
        return;
      }
      --(*coords);
    }
    //! Finishes the underlying loop of 'decrement_coords'.
    template <typename Index>
    static inline void decrement_coords_(Index *coords, const Index *size,
                                         bool& stop)
    {
      if (*coords == 0)
      {
        stop = true;
        return;
      }
      --(*coords);
    }
    //! Finishes the incremented coordinates.
    template <typename Index>
    static inline void decrement_coords(Index *coords,
                                        const Index *start, const Index *end,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      if (*coords == 0)
      {
        stop = true;
        return;
      }
      --(*coords);
    }
    //! Finishes the underlying loop of 'increment_coords'.
    template <typename Index>
    static inline void decrement_coords_(Index *coords,
                                         const Index *start, const Index *end,
                                         bool& stop)
    {
      if (*coords == *start)
      {
        stop = true;
        return;
      }
      --(*coords);
    }
  };

  //! \brief The specialized offset computer for dimension N > 1 and 
  //! column-major storage.
  template <int N>
  struct Offset<N, ColMajor>
  {
    //! Compile-time instantiation check.
    DO_STATIC_ASSERT(N > 0, N_MUST_BE_POSITIVE);
    //! Computes the index that corresponds to the coordinates of an ND array by
    //! loop unrolling.
    template <typename Index>
    static inline Index eval(const Index *coords, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return (*coords) + (*size)*Offset<N-1, ColMajor>::eval(coords+1, size+1);
    }
    //! Computes the index that corresponds to the coordinates of a 2D array.
    template <typename Index>
    static inline Index eval(Index i, Index j, Index size1, Index size2)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return i + j*size1;
    }
    //! Computes the index that corresponds to the coordinates of a 3D array.
    template <typename Index>
    static inline Index eval(Index i, Index j, Index k,
                             Index size1, Index size2, Index size3)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return i + size1 * (j + k*size2);
    }
    //! Computes the strides from the sizes of an ND array.
    template <typename Index>
    static inline void eval_strides(Index *strides, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      strides[0] = 1;
      for (int i = 1; i < N; ++i)
        strides[i] = strides[i-1]*size[i-1];
    }
    //! Computes the coordinates from the index.
    template <typename Index>
    static inline void eval_coords_from_offset(Index *coords, Index offset,
                                               const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      /*for (int i = 0; i < N-1; ++i)
      {
        coords[i] = offset % dims[i];
        offset = (offset - coords[i])/dims[i];
      }
      coords[N-1] = (offset - coords[N-2]) / dims[N-2];*/
      Offset<N, ColMajor>::eval_coords_from_offset_(coords, offset, dims);
    }
    //! Unrolls the underlying loop of 'eval_coords_from_offset'
    template <typename Index>
    static inline void eval_coords_from_offset_(Index *coords, 
                                                Index offset,
                                                const Index *dims)
    {
      *coords = offset % *dims;
      offset = (offset - *coords) / *dims;
      ++coords; ++dims;
      Offset<N-1, ColMajor>::eval_coords_from_offset_(coords, offset, dims);
    }
    //! Computes the incremented coordinates.
    template <typename Index>
    static inline void increment_coords(Index *coords, const Index *size,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);        
//      for (int axis = 0; axis < N; ++axis, ++coords, ++size)
//      {
//        ++(*coords);
//        if (*coords < *size)
//          break;
//        *coords = 0;
//      }
      Offset<N, ColMajor>::increment_coords_(coords, size, stop);
    }
    //! Unrolls the underlying loop of 'increment_coords'.
    template <typename Index>
    static inline void increment_coords_(Index *coords, const Index *size,
                                         bool& stop)
    {
      ++(*coords);
      if (*coords < *size)
        return;
      *coords = 0;
      ++coords; ++size;
      Offset<N-1, ColMajor>::increment_coords_(coords, size, stop);
    }
    //! Computes the incremented coordinates.
    template <typename Index>
    static inline void increment_coords(Index *coords,
                                        const Index *start, const Index *end,
                                        bool& stop)
    {
      Offset<N, ColMajor>::increment_coords_(coords, start, end, stop);
    }
    //! Unrolls the underlying loop of 'increment_coords'.
    template <typename Index>
    static inline void increment_coords_(Index *coords,
                                         const Index *start, const Index *end,
                                         bool& stop)
    {
      // precondition:
      // *start <= *coords && *coords < *end
      ++(*coords);
      if (*coords != *end)
        return;
      *coords = *start;
      ++coords; ++start; ++end;
      Offset<N-1, ColMajor>::increment_coords_(coords, start, end, stop);
    }
    //! Computes the decremented coordinates.
    template <typename Index>
    static inline void decrement_coords(Index *coords, const Index *size,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
//      for (int axis = 0; axis < N; ++axis, ++coords, ++size)
//      {
//        if (*coords > 0)
//        {
//          --(*coords);
//          break;
//        }
//        *coords = *size-1;
//      }
      Offset<N, ColMajor>::decrement_coords_(coords, size, stop);
    }
    //! Unrolls the underlying loop of 'decrement_coords'.
    template <typename Index>
    static inline void decrement_coords_(Index *coords, const Index *size,
                                         bool& stop)
    {
      if (*coords > 0)
      {
        --(*coords);
        return;
      }
      *coords = *size - 1;
      ++coords; ++size;
      Offset<N-1, ColMajor>::decrement_coords_(coords, size, stop);
    }
    //! Computes the incremented coordinates.
    template <typename Index>
    static inline void decrement_coords(Index *coords,
                                        const Index *start, const Index *end,
                                        bool& stop)
    {
      Offset<N, ColMajor>::decrement_coords_(coords, start, end, stop);
    }
    //! Unrolls the underlying loop of 'increment_coords'.
    template <typename Index>
    static inline void decrement_coords_(Index *coords,
                                         const Index *start, const Index *end,
                                         bool& stop)
    {
      // precondition:
      // *start <= *coords && *coords < *end
      if (*coords > *start)
      {
        --(*coords);
        return;
      }
      *coords = *end-1;
      ++coords; ++start; ++end;
      Offset<N-1, ColMajor>::decrement_coords_(coords, start, end, stop);
    }
  };

  //! \brief The specialized offset computer for dimension 1 and 
  //! column-major storage.
  template <>
  struct Offset<1, ColMajor>
  {
    //! Merely returns the 1D coordinates.
    template <typename Index>
    static inline Index eval(const Index *coords, const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      return *coords; 
    }
    //! Merely returns 1.
    template <typename Index>
    static inline void eval_strides(Index *strides, const Index *size)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      *strides = 1;
    }
    //! Merely returns the offset value.
    template <typename Index>
    static inline void eval_coords_from_offset(Index *coords, Index offset,
                                               const Index *dims)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      *coords = offset;
    }
    //! Finishes the unrolled loop of 'eval_coords_from_offset'.
    template <typename Index>
    static inline void eval_coords_from_offset_(Index *coords, Index offset,
                                                const Index *dims)
    {
      *coords = (offset - *(coords-1)) / *(dims-1);
    }
    //! Merely does +1.
    template <typename Index>
    static inline void increment_coords(Index *coords, const Index *size,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      if (*coords == *size-1)
      {
        stop = true;
        return;
      }
      ++(*coords);
    }
    //! Finishes the underlying loop of 'increment_coords'.
    template <typename Index>
    static inline void increment_coords_(Index *coords, const Index *size,
                                         bool& stop)
    {
      if (*coords == *size-1)
      {
        stop = true;
        return;
      }
      ++(*coords);
    }
    //! Merely does +1.
    template <typename Index>
    static inline void increment_coords(Index *coords,
                                        const Index *start, const Index *end,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      if (*coords == *end-1)
      {
        stop = true;
        return;
      }
      ++(*coords);
    }
    //! Finishes the underlying loop of 'increment_coords'.
    template <typename Index>
    static inline void increment_coords_(Index *coords,
                                         const Index *start, const Index *end,
                                         bool& stop)
    {
      // precondition:
      // *start <= *coords && *coords < *end
      if (*coords == *end-1)
      {
        stop = true;
        return;
      }
      ++(*coords);
    }
    //! Merely does -1.
    template <typename Index>
    static inline void decrement_coords(Index *coords, const Index *size,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      if (*coords == 0)
      {
        stop = true;
        return;
      }
      --(*coords);
    }
    //! Finishes the underlying loop of 'decrement_coords'.
    template <typename Index>
    static inline void decrement_coords_(Index *coords, const Index *size,
                                         bool& stop)
    {
      if (*coords == 0)
      {
        stop = true;
        return;
      }
      --(*coords);
    }
    //! Merely does -1.
    template <typename Index>
    static inline void decrement_coords(Index *coords,
                                        const Index *start, const Index *end,
                                        bool& stop)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      if (*coords == *start)
      {
        stop = true;
        return;
      }
      --(*coords);
    }
    //! Finishes the underlying loop of 'decrement_coords'.
    template <typename Index>
    static inline void decrement_coords_(Index *coords,
                                         const Index *start, const Index *end,
                                         bool& stop)
    {
      // precondition:
      // *start <= *coords && *coords < *end
      if (*coords == *start)
      {
        stop = true;
        return;
      }
      --(*coords);
    }
  };

  template <int N>
  struct Offset2
  {
    template <typename Pointer, typename Index>
    static inline void advance(Pointer& p, const Index *strides, const Index *t)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      p += *strides++ * *t++;
      Offset2<N-1>::advance(p, strides, t);
    }
    template <typename Pointer, typename Index>
    static inline void reverse(Pointer& pos, const Index *strides, const Index *t)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      pos -= *strides++ * *t++;
      Offset2<N-1>::reverse(pos, strides, t);
    }
    template <typename Index>
    static inline Index eval(const Index *coords, const Index *strides)
    {
      return (*coords)*(*strides)+Offset2<N-1>::eval(coords+1, strides+1);
    }
  };

  template <>
  struct Offset2<1>
  {
    template <typename Pointer, typename Index>
    static inline void advance(Pointer& ptr, const Index *strides, const Index *t)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      ptr += *strides * *t;
    }
    template <typename Pointer, typename Index>
    static inline void reverse(Pointer& ptr, const Index *strides, const Index *t)
    {
      DO_STATIC_ASSERT(std::numeric_limits<Index>::is_integer,
        INDEX_MUST_BE_INTEGRAL);
      ptr -= *strides * *t;
    }
    template <typename Index>
    static inline Index eval(const Index *coords, const Index *strides)
    {
      return (*coords)*(*strides);
    }
  };

#define ITERATOR_BASE_TYPE(IsConst)                                 \
  std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t, \
    typename Meta::Choose<IsConst, const T *, T *>::Type,           \
    typename Meta::Choose<IsConst, const T&, T&>::Type>

#define TYPEDEF_ITERATOR_TYPES(IteratorType)                      \
  typedef typename base_type::value_type value_type;              \
  typedef typename base_type::difference_type difference_type;    \
  typedef typename base_type::pointer pointer;                    \
  typedef typename base_type::reference reference;                \
  typedef typename base_type::iterator_category iterator_category

  //! \brief Axis iterator class for N-dimensional arrays.
  template <bool IsConst, typename T, int Axis, int N>
  class AxisIterator : public ITERATOR_BASE_TYPE(IsConst)
  {
    DO_STATIC_ASSERT(Axis >= 0 && Axis < N,
      AXIS_MUST_BE_NONNEGATIVE_AND_LESS_THAN_N);
    typedef ITERATOR_BASE_TYPE(IsConst) base_type;

  public: /* STL-like typedefs */
    TYPEDEF_ITERATOR_TYPES(base_type);
    typedef AxisIterator self_type;
    typedef Matrix<int, N, 1> coords_type, vector_type;
    typedef coords_type& coords_reference;
    typedef pointer& pointer_reference;

  public: /* interface */
    //! Constructor.
    inline AxisIterator(pointer_reference pos, coords_reference coords, 
                        const vector_type& strides, const vector_type& size)
      : pos_(pos), coords_(coords), strides_(strides), sizes_(size) {}
    //! Copy constructor.
    inline AxisIterator(const self_type& it)
      : pos_(it.pos_), coords_(it.coords_)
      , strides_(it.strides_), sizes_(it.sizes_) {}

  public: /* dereferencing, access functions. */
    //! Dereferencing operator.
    inline reference operator*() const
    { return *pos_; }
    //! Dereferencing operator.
    inline pointer operator->() const
    { return pos_; }
    //! Access operator.
    inline reference operator[](int n) const
    {
      if (coords_[Axis]+n >= 0  && coords_[Axis]+n < sizes_[Axis])
        return *(pos_+strides_[Axis]*n);
      return *pos_;
    }

  public: /* comparison functions. */
    //! Equality operator.
    template <typename Iterator>
    inline bool operator==(const Iterator& rhs) const
    { return pos_ == rhs.operator->(); }
    //! Equality operator.
    inline bool operator==(pointer pos) const
    { return pos_ == pos; }
    //! Inequality operator.
    template <typename Iterator>
    inline bool operator!=(const Iterator& rhs) const
    { return pos_ != rhs.operator->(); }
    //! Inequality operator.
    inline bool operator!=(pointer pos) const
    { return pos_ != pos; }

  public: /* iterator functionalities. */
    //! Prefix increment operator.
    inline self_type& operator++()
    {
      if (coords_[Axis]+1 < sizes_[Axis])
      {
        pos_ += strides_[Axis];
        ++coords_[Axis];
      }
      return *this;
    }
    //! Prefix decrement operator.
    inline self_type& operator--()
    {
      if (coords_[Axis]-1 >= 0)
      {
        pos_ -= strides_[Axis];
        --coords_[Axis];
      }
      return *this;
    }
    //! Postfix increment operator.
    inline self_type operator++(int)
    { AxisIterator old(*this); operator++(); return old; }
    //! Postfix decrement operator.
    inline self_type operator--(int)
    { AxisIterator old(*this); operator--(); return old; }
    //! Arithmetic operator.
    inline self_type& operator+=(int n)
    {
      if (coords_[Axis]+n >= 0  && coords_[Axis]+n < sizes_[Axis])
      {
        pos_ += strides_[Axis]*n;
        coords_[Axis] += n;
      }
      return *this;
    }
    //! Arithmetic operator.
    inline self_type& operator-=(int n)
    {
      if (coords_[Axis]-n >= 0  && coords_[Axis]-n < sizes_[Axis])
      {
        pos_ -= strides_[Axis]*n;
        coords_[Axis] -= n;
      }
      return *this;
    }

  public: /* additional functionalities. */
    //! Constant size accessor.
    inline int size() const { return sizes_[Axis]; }

  private: /* data members. */
    pointer_reference pos_; //!< current pointer.
    coords_reference coords_; //!< current coordinates.
    const vector_type& strides_; //!< strides.
    const vector_type& sizes_; //!< sizes.
  };

  //! \brief N-dimensional iterator class.
  //! In any case the 'Locator2' class is a heavy object. It is slower 
  //! than the 'Locator' class for incremental iteration.
  //! It is mostly useful for differential calculus. Otherwise prefer using 
  //! other iterator instead.
  template <bool IsConst, typename T, int N, int StorageOrder = ColMajor>
  class RangeIteratorBase : public ITERATOR_BASE_TYPE(IsConst)
  {
    typedef Offset<N, StorageOrder> offset;
    typedef ITERATOR_BASE_TYPE(IsConst) base_type;

  public: /* typedefs */
    TYPEDEF_ITERATOR_TYPES(base_type);
    typedef RangeIteratorBase self_type;
    typedef Matrix<int, N, 1> coords_type, vector_type;
    typedef AxisIterator<IsConst, value_type, 0, N> x_iterator;
    typedef AxisIterator<IsConst, value_type, 1, N> y_iterator;
    typedef AxisIterator<IsConst, value_type, 2, N> z_iterator;

  public: /* interface */
    //! Constructor
    inline RangeIteratorBase(pointer pos,
                             const coords_type& coords,
                             const vector_type& sizes,
                             const vector_type& strides,
                             bool stop)
      : base_type()
      , cur_pos_(pos), cur_coords_(coords)
      , sizes_(sizes), strides_(strides)
      , stop_(stop) {}
    //! Copy constructor
    inline RangeIteratorBase(const self_type& l)
      : base_type()
      , cur_pos_(l.cur_pos_), cur_coords_(l.cur_coords_)
      , sizes_(l.sizes_), strides_(l.strides_)
      , stop_(l.stop_) {}

  public: /* dereferencing functions. */
    //! Dereferencing operator.
    inline reference operator*() const
    { return *cur_pos_; }
    //! Dereferencing operator.
    inline pointer operator->() const
    { return cur_pos_; }
    //! Special dereferencing operator.
    inline reference operator()(int i, int j) const
    { return *(cur_pos_ + strides_[0]*i + strides_[1]*j); }
    //! Special dereferencing operator.
    inline reference operator()(int i, int j, int k) const
    { return *(cur_pos_ + strides_[0]*i + strides_[1]*j + strides_[2]*k); }
    //! Special dereferencing operator.
    inline reference operator()(const vector_type& t) const 
    {
      pointer pos = cur_pos_;
      Offset2<N>::advance(pos, strides_.data(), t.data());
      return *pos;
    }
    //! Special dereferencing operator (mostly for the hessian matrix).
    inline reference delta(int i, int di, int j, int dj) const
    { return *(cur_pos_ + strides_[i]*di + strides_[j]*dj); }
    //! Special dereferencing operator (mostly for the hessian matrix).
    template<int I, int J>
    inline reference delta(int di, int dj) const
    {
      DO_STATIC_ASSERT(I >= 0 && I < N && J >= 0 && J < N,
        I_AND_J_MUST_BETWEEN_0_AND_N);
      return *(cur_pos_ + strides_[I]*di + strides_[J]*dj);
    }

    //! Axis iterator getter.
    //! The axes matches with the Cartesian view if the data is stored in a
    //! row major fashion.
    template <int Axis>
    inline AxisIterator<IsConst, T, Axis, N> axis()
    {
      return AxisIterator<IsConst, T, Axis, N>(
        cur_pos_, cur_coords_, strides_, sizes_); 
    }
    //! X-axis iterator getter.
    inline x_iterator x() { return axis<0>(); }
    //! Y-axis iterator getter.
    inline y_iterator y() { return axis<1>(); }
    //! Z-Axis iterator getter.
    inline z_iterator z() { return axis<2>(); }

  public: /* comparison functions */
    //! Equality operator.
    inline bool operator==(const self_type& rhs) const 
    { return stop_ ? rhs.stop_ : (!rhs.stop_ && cur_pos_ == rhs.cur_pos_); }
    //! Equality operator.
    inline bool operator==(pointer pos) const
    { return cur_pos_ == pos; }
    //! Inequality operator.
    inline bool operator!=(const self_type& rhs) const
    { return !this->operator==(rhs); }
    //! Inequality operator.
    inline bool operator!=(pointer pos) const
    { return !this->operator==(pos); }

  public: /* additional features. */
    //! Get the current coordinates.
    inline const coords_type& coords() const
    { return cur_coords_; }
    //! Get the sizes.
    inline const vector_type& sizes() const
    { return sizes_; }
    //! Get the size of the i-th dimension.
    inline int size(int i) const
    { return sizes_[i]; }
    //! Debugging method.
    inline void check() const
    {
      std::cout << "Checking locator" << std::endl;
      std::cout << "Coords = " << cur_coords_.transpose() << std::endl;
      std::cout << "Value = " << std::endl << *cur_pos_ << std::endl;
    }
    //! Debugging method.
    inline void check_strides() const
    {
      std::cout << "Checking locator strides" << std::endl;
      std::cout << "Strides = " << strides_.transpose() << std::endl;
      std::cout << "Sizes = " << sizes_.transpose() << std::endl;
    }

  protected: /* data members */
    pointer cur_pos_; //!< current pointer.
    coords_type cur_coords_; //!< current coordinates.
    const vector_type& sizes_; //!< sizes.
    const vector_type& strides_; //!< strides.
    bool stop_;
  };

  //! \brief N-dimensional iterator class.
  //! In any case the 'RangeIterator' class is a heavy object.
  //! It is mostly useful for differential calculus. Otherwise prefer using 
  //! other iterator instead.
  template <bool IsConst, typename T, int N, int StorageOrder = ColMajor>
  class RangeIterator : public RangeIteratorBase<IsConst, T, N, StorageOrder>
  {
    typedef Offset<N, StorageOrder> offset;
    typedef RangeIteratorBase<IsConst, T, N, StorageOrder> base_type;
    using base_type::cur_pos_;
    using base_type::cur_coords_;
    using base_type::sizes_;
    using base_type::strides_;
    using base_type::stop_;
  
  public: /* typedefs */
    TYPEDEF_ITERATOR_TYPES(base_type);
    typedef RangeIterator self_type;
    typedef Matrix<int, N, 1> coords_type, vector_type;

  public: /* constructors. */
    //! Constructor
    inline RangeIterator(pointer pos, const coords_type& coords,
                         pointer first, pointer last,
                         const vector_type& sizes,
                         const vector_type& strides,
                         bool stop = false)
      : base_type(pos, coords, sizes, strides, stop)
      , first_(first), last_(last)
    {}
    //! Copy constructor
    inline RangeIterator(const self_type& l)
      : base_type(l.cur_pos_, l.cur_coords_, l.sizes_, l.strides_, l.stop_)
      , first_(l.first_), last_(l.last_)
    {}

  public: /* iterator functionalities. */
    //! Prefix increment operator.
    inline self_type& operator++()
    {
      ++cur_pos_;
      offset::increment_coords(cur_coords_.data(), sizes_.data(), stop_);
      return *this;
    }
    //! Prefix decrement operator.
    inline self_type& operator--()
    {
      --cur_pos_;
      offset::decrement_coords(cur_coords_.data(), sizes_.data(), stop_);
      return *this;
    }
    //! Postfix increment operator (slow: avoid using it).
    inline self_type operator++(int)
    { self_type old(*this); operator++(); return old; }
    //! Postfix decrement operator (slow: avoid using it).
    inline self_type operator--(int)
    { self_type old(*this); operator--(); return old; }
    //! Arithmetic operator (slow: avoid using it).
    inline self_type& operator+=(const vector_type& t)
    {
      /*for (int i = 0; i < N; ++i)
        pos_ += strides_[i]*t[i];*/
      if ((sizes_ - cur_coords_ - t).minCoeff() > 0)
      {
        Offset2<N>::advance(cur_pos_, strides_.data(), t.data());
        cur_coords_ += t;
      }
      return *this;
    }
    //! Arithmetic operator (slow: avoid using it)
    inline self_type& operator-=(const vector_type& t)
    {
      /*for (int i = 0; i < N; ++i)
        pos_ -= strides_[i]*t[i];*/
      if ((cur_coords_ - t).minCoeff() >= 0)
      {
        Offset2<N>::reverse(cur_pos_, strides_.data(), t.data());
        cur_coords_ -= t;
      }
      return *this;
    }

  public: /* additional features. */
    inline void reset_anchor(const coords_type& c = coords_type::Zero())
    {
      pointer pos = first_ + offset::eval(c.data(), sizes_.data());
      if (first_ <= pos && pos < last_)
      {
        cur_pos_ = first_ + offset::eval(c.data(), sizes_.data());
        cur_coords_ = c;
        stop_ = false;
      }
    }
    inline void reset_anchor(int x, int y)
    { reset_anchor(coords_type(x,y)); }
    inline void reset_anchor(int x, int y, int z)
    { reset_anchor(coords_type(x,y,z)); }

  protected:
    pointer first_, last_;
  };

  //! \brief N-dimensional iterator class.
  //! In any case the 'SubrangeIterator' class is a heavy object. It is slower 
  //! than the 'RangeIterator' class for incremental iteration.
  //! It is mostly useful for differential calculus. Otherwise prefer using 
  //! other iterator instead.
  template <bool IsConst, typename T, int N, int StorageOrder = ColMajor>
  class SubrangeIterator : public RangeIteratorBase<IsConst,T,N,StorageOrder>
  {
    typedef Offset<N, StorageOrder> offset;
    typedef RangeIteratorBase<IsConst, T, N, StorageOrder> base_type;
    using base_type::cur_pos_;
    using base_type::cur_coords_;
    using base_type::sizes_;
    using base_type::strides_;
    using base_type::stop_;

  public: /* typedefs. */
    TYPEDEF_ITERATOR_TYPES(base_type);
    typedef SubrangeIterator self_type;
    typedef Matrix<int, N, 1> coords_type, vector_type;

  public: /* constructors */
    //! Constructor
    inline SubrangeIterator(pointer data, const coords_type& coords,
                            const vector_type& start, const vector_type& end,
                            bool stop,
                            pointer pos,
                            const vector_type& sizes,
                            const vector_type& strides)
      : base_type(pos, coords, sizes, strides, stop)
      , data_(data), start_(start), end_(end)
    {}
    //! Copy constructor
    inline SubrangeIterator(const self_type& l)
      : base_type(l.cur_pos_, l.cur_coords_,
                  l.sizes_, l.strides_,
                  l.stop_)
      , data_(l.data_), start_(l.start_), end_(l.end_)
    {}

  public: /* iterator functionalities. */
    //! Prefix increment operator.
    inline self_type& operator++()
    {
      offset::increment_coords(
        cur_coords_.data(), start_.data(), end_.data(), stop_);
      cur_pos_ = data_ + offset::eval(cur_coords_.data(), sizes_.data());
      return *this;
    }
    //! Prefix decrement operator.
    inline self_type& operator--()
    {
      offset::decrement_coords(
        cur_coords_.data(), start_.data(), end_.data(), stop_);
      cur_pos_ = data_ + offset::eval(cur_coords_.data(), sizes_.data());
      return *this;
    }
    //! Postfix increment operator (slow: avoid using it).
    inline self_type operator++(int)
    { self_type old(*this); operator++(); return old; }
    //! Postfix increment operator (slow: avoid using it).
    inline self_type operator--(int)
    { self_type old(*this); operator--(); return old; }
    //! Arithmetic operator (slow: avoid using it).
    inline self_type& operator+=(const vector_type& t)
    {
      if ((end_ - cur_coords_ - t).minCoeff() > 0)
      {
        cur_coords_ += t;
        Offset2<N>::advance(cur_pos_, strides_.data(), t.data());
      }
      return *this;
    }
    //! Arithmetic operator (slow: avoid using it).
    inline self_type& operator-=(const vector_type& t)
    {
      if ((start_ - t).minCoeff() >= 0)
      {
        cur_coords_ -= t;
        Offset2<N>::reverse(cur_pos_, strides_.data(), t.data());
      }
      return *this;
    }

  protected: /* data members. */
    pointer data_;
    vector_type start_;
    vector_type end_;
  };

#undef ITERATOR_BASE_TYPE
#undef TYPEDEF_ITERATOR_TYPES

  //! \brief Coords iterator class for N-dimensional array.
  //! See if loop unrolling makes them faster.
  template <int N>
  class CoordsIterator
  {
  public:
    typedef CoordsIterator self_type;
    typedef Matrix<int, N, 1> coords_type;

  protected:
    coords_type a_;
    coords_type b_;
    coords_type pos_;
    bool stop_;
  public:
    inline CoordsIterator()
      : stop_(true) {}

    inline CoordsIterator(const coords_type& a, const coords_type& b)
      : a_(a), b_(b), pos_(a), stop_(false) {}

    inline CoordsIterator& operator=(const self_type& it)
    {
      a_ = it.a_;
      b_ = it.b_;
      pos_ = it.pos_;
      stop_ = it.stop_;
      return *this;
    }

    inline bool operator==(const self_type& it) const
    { return (stop_ ? it.stop_ : !it.stop_ && pos_ == it.pos_); }

    inline bool operator!=(const self_type& it) const
    { return (stop_ ? !it.stop_ : it.stop_ || pos_ != it.pos_); }

    inline self_type& operator++()
    {
      for (int i=0;i<N;i++) {
        if (pos_[i]!=b_[i]) {
          pos_[i]++;
          return *this;
        }
        pos_[i]=a_[i];
      }
      stop_ = true;
      return *this;
    }

    inline self_type operator++(int)
    { CoordsIterator tmp(*this); ++(*this); return tmp; }

    inline coords_type operator*() const
    { return pos_; }

    inline const coords_type* operator->() const
    { return &pos_; }
  };

  //! @}

} /* namespace DO */

#endif /* DO_CORE_LOCATOR_HPP */
