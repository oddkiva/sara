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


namespace DO::Shakti::HalideBackend::v3 {

  //! @brief List of extrema localized in the discretized scale-space.
  struct QuantizedExtremumArray
  {
    //! @brief Quantized localization.
    Halide::Runtime::Buffer<std::int32_t> x;
    Halide::Runtime::Buffer<std::int32_t> y;
    Halide::Runtime::Buffer<std::int32_t> s;
    Halide::Runtime::Buffer<std::int32_t> n;
    Halide::Runtime::Buffer<float> scale;
    //! @}

    //! @brief Extremum type.
    Halide::Runtime::Buffer<std::int8_t> type;

    QuantizedExtremumArray() = default;

    QuantizedExtremumArray(std::size_t size)
    {
      resize(size);
    }

    struct View
    {
      std::int32_t& x;
      std::int32_t& y;
      std::int32_t& s;
      std::int32_t& n;
      float& scale;
      std::int8_t& type;
    };

    struct ConstView
    {
      const std::int32_t& x;
      const std::int32_t& y;
      const std::int32_t& s;
      const std::int32_t& n;
      const float& scale;
      const std::int8_t& type;
    };

    auto operator[](int i) -> View
    {
      return {x(i), y(i), s(i), n(i), scale(i),type(i)};
    }

    auto operator[](int i) const -> ConstView
    {
      return {x(i), y(i), s(i), n(i), scale(i), type(i)};
    }

    auto resize(std::size_t size) -> void
    {
      x = Halide::Runtime::Buffer<std::int32_t>(size);
      y = Halide::Runtime::Buffer<std::int32_t>(size);
      s = Halide::Runtime::Buffer<std::int32_t>(size);
      n = Halide::Runtime::Buffer<std::int32_t>(size);
      scale = Halide::Runtime::Buffer<float>(size);
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
    Halide::Runtime::Buffer<std::int32_t> n;
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
      std::int32_t& n;
      float& value;
      std::int8_t& type;
    };

    struct ConstView
    {
      const float& x;
      const float& y;
      const float& s;
      const std::int32_t& n;
      const float& value;
      const std::int8_t& type;
    };

    auto operator[](int i) -> View
    {
      return {x(i), y(i), s(i), n(i), value(i), type(i)};
    }

    auto operator[](int i) const -> ConstView
    {
      return {x(i), y(i), s(i), n(i), value(i), type(i)};
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
      n = Halide::Runtime::Buffer<std::int32_t>(size);
      value = Halide::Runtime::Buffer<float>(size);
      type = Halide::Runtime::Buffer<std::int8_t>(size);
    }
  };

}  // namespace DO::Shakti::HalideBackend::v3
