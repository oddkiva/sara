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

  //! Rudimentary polynomial class.
  template <typename T, int N>
  class MatrixBasedObject
  {
  public:
    enum { Dimension = N };
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


  template <typename T, int N>
  class Homography : public MatrixBasedObject<T,N>
  {
    using base_type = MatrixBasedObject<T, N>;
    using base_type::_m;

  public:
    using base_type::Dimension;
    using matrix_type = typename base_type::matrix_type;
    using homogeneous_vector_type = typename base_type::homogeneous_vector_type;
    using euclidean_vector_type = typename base_type::euclidean_vector_type;

    //! @{
    //! @brief Common constructors
    Homography() = default;

    inline Homography(const base_type& other)
      : base_type{other}
    {
    }

    inline Homography(const matrix_type& data)
      : base_type{data}
    {
    }
    //! @}

    //! @{
    //! @brief Evaluation at point 'x'.
    inline auto operator()(const homogeneous_vector_type& x) const
        -> homogeneous_vector_type
    {
      homogeneous_vector_type h_x = (_m * x);
      h_x /= h_x(2);
      return h_x;
    }

    inline auto operator()(const euclidean_vector_type& x) const
        -> euclidean_vector_type
    {
      return (*this)((homogeneous_vector_type{} << x, 1).finished());
    }
    //! @}
  };


} /* namespace Sara */
} /* namespace Projective */
} /* namespace DO */
