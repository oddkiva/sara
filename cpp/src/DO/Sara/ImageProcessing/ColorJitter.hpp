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


namespace DO { namespace Sara {


  struct NormalDistribution
  {
    NormalDistribution()
    {
    }

    float operator()() const
    {
      return _dist(_gen);
    }

    mutable std::mt19937 _gen;
    mutable std::normal_distribution<float> _dist;
  };


  template <int N>
  Image<float, N> normal(const Matrix<int, N, 1>& sizes)
  {
    auto image = Image<float, N>{sizes};
    auto dice = NormalDistribution{};
    for (int i = 0; i < image.size(); ++i)
      image.data()[i] = dice();
    return image;
  }



} /* namespace Sara */
} /* namespace DO */
