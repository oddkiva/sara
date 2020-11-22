// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  template <typename T>
  struct Interval
  {
    T a;
    T b;

    inline auto operator<(const Interval& other) const
    {
      if (a < other.a)
        return true;
      if (a == other.a && b < other.b)
        return true;
      return false;
    }

    inline auto operator==(const Interval& other) const
    {
      return a == other.a && b == other.b;
    }
  };

  struct Cluster1D {
    std::size_t peak;
    Interval<std::size_t> interval;

    auto operator==(const Cluster1D& other) const
    {
      return peak == other.peak && interval == other.interval;
    }
  };

  template <typename T>
  auto compute_density_based_clustering_1d(       //
      const T* density_function,                  //
      std::size_t size,                           //
      T eps = std::numeric_limits<T>::epsilon())  //
      -> std::vector<Cluster1D>
  {
    if (size < 3)
      throw std::runtime_error{"The density function must have size >= 3"};

    auto clusters = std::vector<Cluster1D>{};

    const auto& f = density_function;

    // Localize the local maxima of the density function.
    for (auto i = 1ul; i < size - 1; ++i)
      if (f[i] > f[i - 1] && f[i] > f[i + 1])
        clusters.push_back({i, {}});

    // Find the left bound of each cluster interval.
    for (auto& cluster : clusters)
    {
      for (auto i = std::ptrdiff_t(cluster.peak) - 1; i >= 0; --i)
      {
        if (i == 0 && f[i] > eps)
        {
          cluster.interval.a = i;
          break;
        }

        if ((f[i] < f[i - 1] && f[i] < f[i + 1]) ||
            (f[i - 1] < eps && f[i + 1] > eps && f[i] < f[i + 1]))
        {
          cluster.interval.a = i;
          break;
        }
      }
    }

    // Find the right bound of each cluster interval.
    for (auto& cluster : clusters)
    {
      for (auto i = cluster.peak + 1; i < size; ++i)
      {
        if (i == size - 1 && f[i] > eps)
        {
          cluster.interval.b = i + 1;
          break;
        }

        if ((f[i] < f[i - 1] && f[i] < f[i + 1]) ||
            (f[i - 1] > eps && f[i + 1] < eps && f[i] < f[i - 1]))
        {
          cluster.interval.b = i + 1;
          break;
        }
      }
    }

    return clusters;
  }
}
