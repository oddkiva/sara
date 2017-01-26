// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_IMAGEPROCESSING_SECONDMOMENTMATRIX_HPP
#define DO_SARA_IMAGEPROCESSING_SECONDMOMENTMATRIX_HPP


#include <DO/Sara/Core/Image/Image.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup Differential
    @{
   */

  /*!
    @brief Second-moment matrix helper class in order to use
    DO::Image<T>::compute<SecondMomentMatrix>()
   */
  struct SecondMomentMatrix
  {
    template <typename GradientField>
    using Scalar = typename GradientField::pixel_type::Scalar;

    template <typename GradientField>
    using Matrix =
        Eigen::Matrix<Scalar<GradientField>, GradientField::Dimension,
                      GradientField::Dimension>;

    template <typename GradientField>
    using MatrixField = Image<Matrix<GradientField>, GradientField::Dimension>;

    template <typename GradientField>
    auto operator()(const GradientField& in) -> MatrixField<GradientField>
    {
      auto out = MatrixField<GradientField>{ in.sizes() };

      auto out_i = out.begin();
      auto in_i = in.begin();
      for ( ; in_i != in.end(); ++in_i, ++out_i)
        *out_i = *in_i * in_i->transpose();

      return out;
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_IMAGEPROCESSING_SECONDMOMENTMATRIX_HPP */
