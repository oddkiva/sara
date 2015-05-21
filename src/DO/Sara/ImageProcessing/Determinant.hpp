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

#ifndef DO_SARA_IMAGEPROCESSING_DETERMINANT_HPP
#define DO_SARA_IMAGEPROCESSING_DETERMINANT_HPP


#include <DO/Sara/Core/Image/Image.hpp>


namespace DO {

  /*!
    \ingroup Differential
    @{
   */

  //! \brief Helper class to use Image<T,N>::compute<Determinant>()
  template <typename Matrix_, int N>
  struct Determinant
  {
    typedef Matrix<int, N, 1> coords_type;
    typedef typename Matrix_::Scalar scalar_type;
    typedef Image<Matrix_, N> matrix_field_type;
    typedef Image<scalar_type, N> scalar_field_type, return_type;

    inline Determinant(const matrix_field_type& matrixField)
      : matrix_field_(matrixField) {}

    scalar_field_type operator()() const
    {
      scalar_field_type det_field_(matrix_field_.sizes());
      typename scalar_field_type::iterator dst = det_field_.begin();
      typename matrix_field_type::const_iterator src = matrix_field_.begin();
      for ( ; src != matrix_field_.end(); ++src, ++dst)
        *dst = src->determinant();
      return det_field_;
    }

    const matrix_field_type& matrix_field_;
  };

  //! @}

} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_DETERMINANT_HPP */