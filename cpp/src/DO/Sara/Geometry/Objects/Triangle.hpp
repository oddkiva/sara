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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Geometry/Objects/Polygon.hpp>


namespace DO { namespace Sara {

  //! @addtogroup GeometryObjects
  //! @{

  // Triangle (a,b,c) enumerated in CCW order.
  class DO_SARA_EXPORT Triangle : public SmallPolygon<3>
  {
  public:
    Triangle() = default;

    Triangle(const Point2d& a, const Point2d& b, const Point2d& c);
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
