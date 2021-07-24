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

#include <DO/Sara/Features/Feature.hpp>


namespace DO { namespace Sara {

  //! @brief Draw the region.
  DO_SARA_EXPORT
  auto draw(const OERegion& f, const Rgb8& c, float scale = 1.f,
            const Point2f& offset = Point2f::Zero()) -> void;

  /*!
   *  @addtogroup Features
   *  @{
   */
  inline auto draw_oe_regions(const OERegion* begin, const OERegion* end,
                              const Rgb8& color, float scale = 1.f,
                              const Point2f& offset = Point2f::Zero())
  {
    std::for_each(begin, end,
                  [&](const auto& f) { draw(f, color, scale, offset); });
  }

  inline auto draw_oe_regions(const std::vector<OERegion>& features,
                              const Rgb8& color, float scale = 1.f,
                              const Point2f& offset = Point2f::Zero())
  {
    draw_oe_regions(features.data(), features.data() + features.size(), color,
                    scale, offset);
  }


  //! @}

}}  // namespace DO::Sara
