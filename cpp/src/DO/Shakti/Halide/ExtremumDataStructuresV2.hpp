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

//! @file

#pragma once

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Shakti/Halide/Utilities.hpp>



namespace DO::Shakti::HalideBackend::v2 {

  //! @brief List of extrema localized in the discretized scale-space.
  struct QuantizedExtremumArray
  {
    //! @brief Quantized localization.
    Halide::Runtime::Buffer<std::int32_t> x;
    Halide::Runtime::Buffer<std::int32_t> y;
    float scale;
    //! @}

    //! @brief Extremum type.
    Halide::Runtime::Buffer<std::int8_t> type;

    QuantizedExtremumArray() = default;

    QuantizedExtremumArray(std::size_t size)
    {
      resize(size);
    }

    auto resize(int size) -> void
    {
      x = Halide::Runtime::Buffer<std::int32_t>(size);
      y = Halide::Runtime::Buffer<std::int32_t>(size);
      type = Halide::Runtime::Buffer<std::int8_t>(size);
    }

    auto empty() const noexcept
    {
      return x.dimensions() == 0;
    }

    auto size() const noexcept
    {
      return x.dim(0).extent();
    }
  };

  //! @brief List of refined extrema in the continuous scale-space.
  struct ExtremumArray
  {
    //! @brief Coordinates of the extrema
    //! @{
    Halide::Runtime::Buffer<float> x;
    Halide::Runtime::Buffer<float> y;
    Halide::Runtime::Buffer<float> s;
    //! @}

    //! @brief Extremum values.
    Halide::Runtime::Buffer<float> value;

    //! @brief Extremum types.
    Halide::Runtime::Buffer<std::int8_t> type;

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
      return {x(i), y(i), s(i), value(i), type(i)};
    }

    auto operator[](int i) const -> ConstView
    {
      return {x(i), y(i), s(i), value(i), type(i)};
    }

    auto empty() const noexcept
    {
      return x.dimensions() == 0;
    }

    auto size() const noexcept
    {
      return x.dim(0).extent();
    }

    auto resize(std::size_t size)
    {
      x = Halide::Runtime::Buffer<float>(size);
      y = Halide::Runtime::Buffer<float>(size);
      s = Halide::Runtime::Buffer<float>(size);
      value = Halide::Runtime::Buffer<float>(size);
      type = Halide::Runtime::Buffer<std::int8_t>(size);
    }
  };


  //! @brief Dominant gradient orientation map for a given list of extrema.
  //! @{
  struct DominantOrientationDenseMap
  {
    Halide::Runtime::Buffer<bool> peak_map;
    Halide::Runtime::Buffer<float> peak_residuals;

    DominantOrientationDenseMap() = default;

    DominantOrientationDenseMap(int num_keypoints,
                                int num_orientation_bins = 36)
    {
      resize(num_keypoints, num_orientation_bins);
    }

    auto resize(int num_keypoints, int num_orientation_bins = 36) -> void
    {
      peak_map =
          Halide::Runtime::Buffer<bool>{num_orientation_bins, num_keypoints};
      peak_residuals =
          Halide::Runtime::Buffer<float>{num_orientation_bins, num_keypoints};
    }

    auto empty() const noexcept
    {
      return peak_map.dimensions() == 0;
    }

    auto num_keypoints() const noexcept
    {
      return peak_map.dim(1).extent();
    }

    auto num_orientation_bins() const noexcept
    {
      return peak_map.dim(0).extent();
    }

    auto copy_to_host()
    {
      peak_map.copy_to_host();
      peak_residuals.copy_to_host();
    }
  };

  struct DominantOrientationSparseMap
  {
    using extremum_index_type = int;
    using angle_type = float;
    using sparse_map_type = std::multimap<extremum_index_type, angle_type>;

    sparse_map_type orientation_map;

    DominantOrientationSparseMap() = default;

    //! @brief Make sure the data is copied to host memory.
    DominantOrientationSparseMap(const DominantOrientationDenseMap& dense)
    {
      const auto peak_map_view =
          Sara::TensorView_<bool, 2>{
              dense.peak_map.data(),
              {dense.num_keypoints(), dense.num_orientation_bins()}}
              .matrix();
      const Eigen::VectorXi peak_count =
          peak_map_view.rowwise().count().cast<int>();

      for (auto k = 0; k < dense.num_keypoints(); ++k)
      {
        if (peak_count(k) == 0)
        {
          orientation_map.insert({k, 0});
          continue;
        }

        const auto N = dense.num_orientation_bins();
        constexpr auto two_pi = 2 * static_cast<float>(M_PI);
        for (auto o = 0; o < dense.num_orientation_bins(); ++o)
        {
          if (!dense.peak_map(o, k))
            continue;

          auto ori = o + dense.peak_residuals(o, k);

          // Make sure that the angle is in the interval [0, N[.
          if (ori < 0)
            ori += N;
          else if (ori > N)
            ori -= N;
          // Convert to radians.
          ori = ori * two_pi / N;

          orientation_map.insert({k, ori});
        }
      }
    }

    operator sparse_map_type&() noexcept
    {
      return orientation_map;
    }

    operator const sparse_map_type&() const noexcept
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
  //! @}


  //! @brief List of oriented extrema.
  struct OrientedExtremumArray : ExtremumArray
  {
    Halide::Runtime::Buffer<float> orientations;

    auto resize(std::int32_t size)
    {
      ExtremumArray::resize(size);
      orientations = Halide::Runtime::Buffer<float>{size};
    }
  };

}  // namespace DO::Shakti::HalideBackend::v2
