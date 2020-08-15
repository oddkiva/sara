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


namespace DO { namespace Sara { namespace Projective {

  //! @addtogroup GeometryTools

  //! TODO: a bit overkill, remove?
  template <typename T, int N>
  class MatrixBasedObject
  {
  public:
    enum
    {
      Dimension = N
    };

    using matrix_type = Matrix<T, N + 1, N + 1>;
    using homogeneous_vector_type = Matrix<T, N + 1, 1>;  // in projective space
    using euclidean_vector_type = Matrix<T, N, 1>;        // in Euclidean space

    //! @{
    //! @brief Common constructors
    MatrixBasedObject() = default;

    inline MatrixBasedObject(const matrix_type& m)
      : _m{m}
    {
    }
    //! @}

    //! @{
    //! @brief Implicit cast operator.
    inline operator matrix_type&()
    {
      return _m;
    }

    inline operator const matrix_type&() const
    {
      return _m;
    }
    //! @}

  protected:
    matrix_type _m;
  };

  //! @}

}}}  // namespace DO::Sara::Projective
