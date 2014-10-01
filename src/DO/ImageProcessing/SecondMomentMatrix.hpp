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

#ifndef DO_IMAGEPROCESSING_SECONDMOMENTMATRIX_HPP
#define DO_IMAGEPROCESSING_SECONDMOMENTMATRIX_HPP


#include <DO/Core/Image/Image.hpp>


namespace DO {

  /*!
    \ingroup Differential
    @{
   */

  /*!
    \brief Second-moment matrix helper class in order to use 
    DO::Image<T>::compute<SecondMomentMatrix>()
   */
  template <typename Vector, int N>
  struct SecondMomentMatrix
  {
    typedef Matrix<int, N, 1> Coords;
    typedef typename Vector::Scalar Scalar;
    typedef Matrix<Scalar, N, N> Moment;
    typedef Image<Vector, N> VectorField;
    typedef Image<Moment, N> MomentField, return_type;

    inline SecondMomentMatrix(const VectorField& gradient_field)
      : gradient_field_(gradient_field) {}
    
    MomentField operator()() const
    {
      MomentField M(gradient_field_.sizes());
      typename MomentField::iterator dst = M.begin();
      typename VectorField::const_iterator src = gradient_field_.begin();
      for ( ; src != gradient_field_.end(); ++src, ++dst)
        *dst = *src * src->transpose();
      return M;
    }

    const VectorField& gradient_field_;
  };

  //! @}

} /* namespace DO */


#endif /* DO_IMAGEPROCESSING_SECONDMOMENTMATRIX_HPP */