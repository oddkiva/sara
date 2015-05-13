// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_CORE_MULTIARRAY_ELEMENTTRAITS_HPP
#define DO_SARA_CORE_MULTIARRAY_ELEMENTTRAITS_HPP


#include <DO/Sara/Core/Meta.hpp>
#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO {

  //! \ingroup MultiArray
  //! @{

  //! The generic traits class for the MultiArray element type.
  //! This traits class is when the array/matrix view is used. This serves as
  //! an interface for the Eigen library.
  template <typename T>
  struct ElementTraits
  {
    //! @{
    //! STL-compatible interface.
    typedef T value_type;
    typedef size_t size_type;
    typedef T * pointer;
    typedef const T * const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T * iterator;
    typedef const T * const_iterator;
    static const bool is_scalar = true;
    //! @}
  };

  //! \brief The specialized element traits class when the entry is a matrix.
  //! Again the matrix is viewed as a scalar. Additions and subtractions between
  //! matrices are OK but multiplication will be the point-wise matrix
  //! multiplication.
  //!
  //! This may be questionable and this may change in the future.
  template <typename T, int M, int N>
  struct ElementTraits<Matrix<T, M, N> >
  {
    //! @{
    //! STL-compatible interface.
    const static bool is_square_matrix = (M == N);
    typedef typename Meta::Choose<
      is_square_matrix,
      Matrix<T, N, N>,
      Array<T, M, N> >::Type value_type;
    typedef size_t size_type;
    typedef value_type * pointer;
    typedef const value_type * const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef value_type * iterator;
    typedef const value_type * const_iterator;
    static const bool is_scalar = false;
    //! @}
  };

  //! \brief The specialized element traits class when the entry is an array.
  //! Default "scalar" operations are point-wise matrix operations.
  template <typename T, int M, int N>
  struct ElementTraits<Array<T, M, N> >
  {
    //! @{
    //! STL-compatible interface.
    typedef Array<T, M, N> value_type;
    typedef size_t size_type;
    typedef value_type * pointer;
    typedef const value_type * const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef value_type * iterator;
    typedef const value_type * const_iterator;
    static const bool is_scalar = false;
    //! @}
  };

  //! @}

}


#endif /* DO_SARA_CORE_MULTIARRAY_ELEMENTTRAITS_HPP */
