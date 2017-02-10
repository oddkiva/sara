// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <random>

#include <DO/Sara/ImageProcessing.hpp>


namespace DO { namespace Sara {


  void add_randn_noise(Image<Rgb32f>& image, float std_dev,
                       const NormalDistribution& dist)
  {
    const auto noise = [&]()
    {
      auto out = Image<Rgb32f>{image.sizes()};
      dist(out);
      return out;
    }();
    image.array() += Vector3f{std_dev, std_dev, std_dev} * noise.array();
  }


} /* namespace Sara */
} /* namespace DO */
