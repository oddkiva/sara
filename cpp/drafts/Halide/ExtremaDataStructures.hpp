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
  struct QuantizedExtremumArray
  {
    //! @brief Quantized localization.
    std::vector<std::int32_t> x;
    std::vector<std::int32_t> y;
    float scale;
    float scale_geometric_factor;
    //! @}

    //! @brief Extremum type.
    std::vector<std::int8_t> type;

    QuantizedExtremumArray() = default;

    QuantizedExtremumArray(std::size_t size)
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
  struct ExtremumArray
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
  struct DominantOrientationDenseMap
  {
    Sara::Tensor_<bool, 2> peak_map;
    Sara::Tensor_<float, 2> peak_residuals;

    DominantOrientationDenseMap() = default;

    DominantOrientationDenseMap(int num_keypoints,
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

  struct DominantOrientationMap
  {
    using extremum_index_type = int;
    using angle_type = float;
    using OrientationMap = std::multimap<extremum_index_type, angle_type>;

    OrientationMap orientation_map;

    operator OrientationMap&() noexcept
    {
      return orientation_map;
    }

    operator const OrientationMap&() const noexcept
    {
      return orientation_map;
    }

    auto dominant_orientations(extremum_index_type i) const
    {
      auto orientations = std::vector<angle_type>{};
      const auto [o_begin, o_end] = orientation_map.equal_range(i);
      for (auto o = o_begin; o != o_end; ++o)
        orientations.push_back(o->second);
      return orientations;
    };
  };


  //! @brief List of oriented extrema in the continuous scale-space.
  struct OrientedExtremumArray : ExtremumArray
  {
    //! @brief Coordinates of the extrema
    //! @{
    std::vector<float> orientations;
    //! @}

    struct View : ExtremumArray::View
    {
      float& orientation;
    };

    struct ConstView : ExtremumArray::ConstView
    {
      const float& orientation;
    };

    auto operator[](int i) -> View
    {
      return {{x[i], y[i], s[i], value[i], type[i]}, orientations[i]};
    }

    auto operator[](int i) const -> ConstView
    {
      return {{x[i], y[i], s[i], value[i], type[i]}, orientations[i]};
    }

    auto resize(std::size_t size)
    {
      ExtremumArray::resize(size);
      orientations.resize(size);
    }
  };


  template <typename T>
  struct Pyramid
  {
    std::map<std::pair<int, int>, std::pair<float, float>> scale_octave_pairs;
    std::map<std::pair<int, int>, T> dict;
  };


  auto to_oriented_extremum_array(const ExtremumArray& extrema,
                                  const DominantOrientationMap& orientations)
  {
    auto oriented_extrema = OrientedExtremumArray{};
    oriented_extrema.scale_quantized = extrema.scale_quantized;

    for (auto e = 0u; e < extrema.size(); ++e)
    {
      const auto ei = extrema[e];
      const auto ois = orientations.dominant_orientations(e);

      for (const auto& oi : ois)
      {
        oriented_extrema.x.push_back(ei.x);
        oriented_extrema.y.push_back(ei.y);
        oriented_extrema.s.push_back(ei.s);
        oriented_extrema.type.push_back(ei.type);
        oriented_extrema.value.push_back(ei.value);
        oriented_extrema.orientations.push_back(oi);
      }
    }

    return oriented_extrema;
  }

  auto to_oriented_extremum_array(
      const Pyramid<ExtremumArray>& extrema,
      const Pyramid<DominantOrientationMap>& orientations)
  {
    auto oriented_extrema = Pyramid<OrientedExtremumArray>{};

    oriented_extrema.scale_octave_pairs = extrema.scale_octave_pairs;

    for (const auto& so : extrema.scale_octave_pairs)
    {
      const auto& s = so.first.first;
      const auto& o = so.first.second;

      auto eit = extrema.dict.find({s, o});
      if (eit == extrema.dict.end())
        continue;

      const auto& extrema_so = eit->second;
      const auto& ori_so = orientations.dict.at({s, o});

      oriented_extrema.dict[{s, o}] = to_oriented_extremum_array(extrema_so,  //
                                                                 ori_so);     //
    }

    return oriented_extrema;
  }

}  // namespace DO::Shakti::HalideBackend
