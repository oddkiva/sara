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

#include <DO/Sara/Geometry/Tools/Metric.hpp>


namespace DO { namespace Sara {

  /*!
   *  @addtogroup FeatureMatching
   *  @{
   */

  //! @brief Functor for geometric filtering purpose.
  class DO_SARA_EXPORT KeyProximity
  {
  public:
    KeyProximity(float metric_dist_thres = .5f, float pixel_dist_thres = 10.f)
      : _squared_metric_dist{metric_dist_thres * metric_dist_thres}
      , _squared_dist_thres{pixel_dist_thres * pixel_dist_thres}
    {
    }

    SquaredRefDistance<float, 2> mapped_squared_metric(const OERegion& f) const
    {
      return SquaredRefDistance<float, 2>(f.shape_matrix);
    }

    bool operator()(const OERegion& f1, const OERegion& f2) const;

  private:
    float _squared_metric_dist;
    float _squared_dist_thres;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
