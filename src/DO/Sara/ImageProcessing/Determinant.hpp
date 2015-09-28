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


namespace DO { namespace Sara {

  /*!
    \ingroup Differential
    @{
   */

  //! \brief Helper class to use Image<T,N>::compute<Determinant>()
  struct Determinant
  {
    template <typename MatrixField>
    using Matrix = typename MatrixField::pixel_type;

    template <typename MatrixField>
    using Scalar = typename Matrix<MatrixField>::Scalar;

    template <typename MatrixField>
    using ReturnType = Image<Scalar<MatrixField>>;

    template <typename MatrixField>
    ReturnType<MatrixField> compute(const MatrixField& in) const
    {
      ReturnType<MatrixField> out(in.sizes());
      auto out_i = out.begin();
      auto in_i = in.begin();
      for ( ; in_i != in.end(); ++in_i, ++out_i)
        *out_i = in_i->determinant();
      return out;
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_DETERMINANT_HPP */
