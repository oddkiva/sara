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

#ifndef DO_SARA_GEOMETRY_QUAD_HPP
#define DO_SARA_GEOMETRY_QUAD_HPP

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>

#include <DO/Sara/Geometry/Objects/Polygon.hpp>
#include <DO/Sara/Geometry/Objects/BBox.hpp>


namespace DO { namespace Sara {

  class DO_SARA_EXPORT Quad : public SmallPolygon<4>
  {
  public:
    //! @{
    //! @brief Constructors.
    Quad(const BBox& bbox);

    Quad(const Point2d& a, const Point2d& b,
         const Point2d& c, const Point2d& d);
    //! @}
  };

} /* namespace Sara */
} /* namespace DO */

#endif /* DO_SARA_GEOMETRY_QUAD_HPP */
