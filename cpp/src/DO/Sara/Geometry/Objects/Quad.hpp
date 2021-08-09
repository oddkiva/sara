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
#include <DO/Sara/Geometry/Objects/BBox.hpp>


namespace DO { namespace Sara {

  //! @addtogroup GeometryObjects
  //! @{

  class Quad : public SmallPolygon<4>
  {
  public:
    //! @{
    //! @brief Constructors.
    DO_SARA_EXPORT
    Quad(const BBox& bbox);

    DO_SARA_EXPORT
    Quad(const Point2d& a, const Point2d& b,
         const Point2d& c, const Point2d& d);
    //! @}
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
