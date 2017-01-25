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

#pragma once

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
    struct Dimension {
      enum { value = GradientField::Dimension };
    };

    template <typename GradientField>
    using Scalar = typename GradientField::pixel_type::Scalar;

    template <typename GradientField>
    using OutPixel =
        Eigen::Matrix<Scalar<GradientField>, Dimension<GradientField>::value,
                      Dimension<GradientField>::value>;

    template <typename GradientField, typename MatrixField>
    void operator()(const GradientField& gradient_field,
                    MatrixField& moment_field) const
    {
      auto dst = moment_field.begin();
      auto src = gradient_field.begin();
      for ( ; src != gradient_field.end(); ++src, ++dst)
        *dst = *src * src->transpose();
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
