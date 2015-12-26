// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_FEATURES_DRAW_HPP
#define DO_SARA_FEATURES_DRAW_HPP

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Features/Feature.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup Features
    @{
  */
  DO_SARA_EXPORT
  void draw_oe_regions(const std::vector<OERegion>& features,
                       const Color3ub& c,
                       float scale = 1.f,
                       const Point2f& off = Point2f::Zero());

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_AFFINECOVARIANTFEATURES_DRAW_H */
