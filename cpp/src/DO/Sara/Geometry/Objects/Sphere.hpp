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


namespace DO { namespace Sara {

  //! @addtogroup GeometryObjects
  //! @{

  class Sphere
  {
    Point3d _c;
    double _r;

  public:
    Sphere(const Point3d& c, double r)
      : _c(c), _r(r)
    {
    }

    const Point3d& center() const
    {
      return _c;
    }

    double radius() const
    {
      return _r;
    }

    bool contains(const Point3d& x) const
    {
      return (x - _c).squaredNorm() < _r*_r;
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
