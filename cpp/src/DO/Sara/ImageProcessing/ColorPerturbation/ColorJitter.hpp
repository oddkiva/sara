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

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/ImageIO.hpp>


namespace DO { namespace Sara {

  
  struct NormalDistribution
  {
    NormalDistribution(std::random_device& rd)
      : _gen(rd)
    {
    }

    float operator()()
    {
      return _dist(_gen);
    }

    std::mt19937 _gen;
    std::normal_distribution<float> _dist;
  };


  template <int N>
  Image<float, N> normal(const Matrix<int, N, 1>& sizes)
  {
    auto random = Image<float, N>{sizes};
    auto dice = NormalDistribution{};
    auto lambda = [&normal_dist](float& value) {
      value = dice();
    };
    std::for_each(random.begin(), random.end(), lambda);
    return random;
  }



} /* namespace Sara */
} /* namespace DO */
