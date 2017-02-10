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

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  class NormalDistribution
  {
  public:
    NormalDistribution(bool seed_randomly = true)
    {
      if (seed_randomly)
        _gen = randomly_seeded_mersenne_twister();
    }

    float operator()() const
    {
      return _dist(_gen);
    }

    template <typename _Matrix>
    inline void operator()(_Matrix& m) const
    {
      std::generate(m.data(), m.data() + m.size(), [&]() {
        return _dist(_gen);
      });
    }

    template <int N>
    void operator()(Image<Rgb32f, N>& image) const
    {
      const auto randn_pixel = [&](Rgb32f& v) {
        v[0] = _dist(_gen);
        v[1] = _dist(_gen);
        v[2] = _dist(_gen);
      };
      image.cwise_transform_inplace(randn_pixel);
    }

  private:
    auto randomly_seeded_mersenne_twister() -> std::mt19937
    {
      constexpr auto N = std::mt19937::state_size;
      unsigned random_data[N];
      std::random_device source;
      std::generate(std::begin(random_data), std::end(random_data),
                    std::ref(source));
      std::seed_seq seeds(std::begin(random_data), std::end(random_data));
      return std::mt19937{seeds};
    }

  private:
    mutable std::mt19937 _gen;
    mutable std::normal_distribution<float> _dist;
  };

  DO_SARA_EXPORT
  void add_randn_noise(Image<Rgb32f>& image, float std_dev,
                       const NormalDistribution& dist);


} /* namespace Sara */
} /* namespace DO */
