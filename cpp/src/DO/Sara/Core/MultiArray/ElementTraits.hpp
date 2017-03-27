// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Meta.hpp>


namespace DO { namespace Sara {

  //! @ingroup MultiArray
  //! @{

  //! The generic traits class for the MultiArray element type.
  //! This traits class is when the array/matrix view is used. This serves as
  //! an interface for the Eigen library.
  template <typename T>
  struct ElementTraits
  {
    //! @{
    //! STL-compatible interface.
    using value_type = T;
    using size_type = size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;
    static const bool is_scalar = true;
    //! @}
  };

  //! @brief The specialized element traits class when the entry is a matrix.
  //! Again the matrix is viewed as a scalar. Additions and subtractions between
  //! matrices are OK but multiplication will be the point-wise matrix
  //! multiplication.
  //!
  //! This may be questionable and this may change in the future.
  template <typename T, int M, int N>
  struct ElementTraits<Matrix<T, M, N>>
  {
    //! @{
    //! STL-compatible interface.
    const static bool is_square_matrix = (M == N);
    using value_type = typename Meta::Choose<is_square_matrix, Matrix<T, N, N>,
                                             Array<T, M, N>>::Type;
    using size_type = size_t;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = value_type*;
    using const_iterator = const value_type*;
    static const bool is_scalar = false;
    //! @}
  };

  //! @brief The specialized element traits class when the entry is an array.
  //! Default "scalar" operations are point-wise matrix operations.
  template <typename T, int M, int N>
  struct ElementTraits<Array<T, M, N>>
  {
    //! @{
    //! STL-compatible interface.
    using value_type = Array<T, M, N>;
    using size_type = size_t;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = value_type *;
    using const_iterator = const value_type *;
    static const bool is_scalar = false;
    //! @}
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
