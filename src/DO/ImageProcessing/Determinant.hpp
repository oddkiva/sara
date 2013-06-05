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

#ifndef DO_IMAGEPROCESSING_DETERMINANT_HPP
#define DO_IMAGEPROCESSING_DETERMINANT_HPP

namespace DO {

  /*!
    \ingroup Differential
    @{
   */

  //! \brief Helper class to use Image<T,N>::compute<Determinant>()
  template <typename Matrix_, int N>
  struct Determinant
  {
    typedef Matrix<int, N, 1> Coords;
    typedef typename Matrix_::Scalar Scalar;
    typedef Image<Matrix_, N> MatrixField;
    typedef Image<Scalar, N> ScalarField, ReturnType;

    inline Determinant(const MatrixField& matrixField)
      : matrix_field_(matrixField) {}

    ScalarField operator()() const
    {
      ScalarField det_field_(matrix_field_.sizes());
      typename ScalarField::iterator dst = det_field_.begin();
      typename MatrixField::const_iterator src = matrix_field_.begin();
      for ( ; src != matrix_field_.end(); ++src, ++dst)
        *dst = src->determinant();
      return det_field_;
    }

    const MatrixField& matrix_field_;
  };

  //! @}

} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_DETERMINANT_HPP */