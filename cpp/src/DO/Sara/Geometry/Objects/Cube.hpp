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

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  //! @addtogroup GeometryObjects
  //! @{

  class Cube
  {
    Point3d _a, _b;

  public:
    Cube() = default;

    Cube(Vector3d& origin, double side)
      : _a{ origin }
      , _b{ (origin.array() + side).matrix() }
    {
    }

    Point3d const& a() const
    {
      return _a;
    }

    Point3d const& b() const
    {
      return _b;
    }

    bool contains(const Point3d& p)
    {
      return p.cwiseMin(_a) == _a && p.cwiseMax(_b) == _b;
    }

    friend double area(const Cube& c)
    {
      return std::pow((c._b - c._a)(0), 3);
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
