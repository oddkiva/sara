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

#include <DO/Sara/Geometry/Tools/MatrixBasedObject.hpp>


namespace DO { namespace Sara { namespace Projective {

  //! @addtogroup GeometryTools
  //! @{

  //! @brief Conic class.
  template <typename T, int N>
  class Conic : public MatrixBasedObject<T,N>
  {
    using Base = MatrixBasedObject<T, N>;
    using Base::_mat;

  public:
    using Base::Dimension;
    using Mat = typename Base::Mat;
    using HVec = typename Base::HVec;
    using Vec = typename Base::Vec;

    //! @{
    //! @brief Common constructors
    Conic() = default;

    inline Conic(const Base& other)
      : Base(other)
    {
    }

    inline Conic(const Mat& data)
      : Base(data)
    {
    }
    //! @}

    //! @{
    //! @brief Evaluation at point 'x'.
    inline T operator()(const HVec& x) const
    {
      return x.transpose() * _mat * x;
    }

    inline T operator()(const Vec& x) const
    {
      return (*this)((HVec() << x, 1).finished());
    }
    //! @}
  };

  //! @brief I/O.
  template <typename T, int N>
  std::ostream& operator<<(std::ostream& os, const Conic<T, N>& P);

  //! @}

} /* namespace Projective */
} /* namespace Sara */
} /* namespace DO */
