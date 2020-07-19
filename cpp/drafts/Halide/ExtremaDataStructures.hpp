// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Shakti::HalideBackend {

  //! @brief List of extrema localized in the discretized scale-space.
  struct QuantizedExtremaArray
  {
    //! @brief Quantized localization.
    std::vector<std::int32_t> x;
    std::vector<std::int32_t> y;
    float scale;
    float scale_geometric_factor;
    //! @}

    //! @brief Extremum type.
    std::vector<std::int8_t> type;

    QuantizedExtremaArray() = default;

    QuantizedExtremaArray(std::size_t size)
    {
      resize(size);
    }

    auto resize(std::size_t size) -> void
    {
      x.resize(size);
      y.resize(size);
      type.resize(size);
    }

    auto size() const
    {
      return x.size();
    }
  };


  //! @brief List of refined extrema in the continuous scale-space.
  struct ExtremaArray
  {
    //! @brief Coordinates of the extrema
    //! @{
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> s;
    //! @}

    //! @brief Extremum values.
    std::vector<float> value;

    //! @brief Extremum types.
    std::vector<std::int8_t> type;

    float scale_quantized;

    struct View
    {
      float& x;
      float& y;
      float& s;
      float& value;
      std::int8_t& type;
    };

    struct ConstView
    {
      const float& x;
      const float& y;
      const float& s;
      const float& value;
      const std::int8_t& type;
    };

    auto operator[](int i) -> View
    {
      return {x[i], y[i], s[i], value[i], type[i]};
    }

    auto operator[](int i) const -> ConstView
    {
      return {x[i], y[i], s[i], value[i], type[i]};
    }

    auto size() const noexcept
    {
      return x.size();
    }

    auto resize(std::size_t size)
    {
      x.resize(size);
      y.resize(size);
      s.resize(size);
      value.resize(size);
      type.resize(size);
    }
  };


  //! @brief Dominant gradient orientation map for a given list of extrema.
  struct DominantGradientOrientationMap
  {
    Sara::Tensor_<bool, 2> peak_map;
    Sara::Tensor_<float, 2> peak_residuals;

    DominantGradientOrientationMap() = default;

    DominantGradientOrientationMap(int num_keypoints,
                                   int num_orientation_bins = 36)
    {
      resize(num_keypoints, num_orientation_bins);
    }

    auto resize(int num_keypoints, int num_orientation_bins = 36) -> void
    {
      peak_map.resize(num_keypoints, num_orientation_bins);
      peak_residuals.resize(num_keypoints, num_orientation_bins);
    }

    auto num_keypoints() const noexcept
    {
      return peak_map.size(0);
    }

    auto num_orientation_bins() const noexcept
    {
      return peak_map.size(1);
    }
  };

  template <typename T>
  struct Pyramid
  {
    std::map<std::pair<int, int>, std::pair<float, float>> scale_octave_pairs;
    std::map<std::pair<int, int>, T> dict;
  };
}
