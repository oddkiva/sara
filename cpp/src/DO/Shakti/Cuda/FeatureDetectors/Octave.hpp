// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/Cuda/MultiArray/CudaArray.hpp>


namespace DO::Shakti::Cuda {

  template <typename T>
  class Octave
  {
  public:
    inline Octave() noexcept = default;

    inline Octave(int width, int height, int scale_count)
      : _array{make_3d_layered_surface_array<T>({width, height, scale_count})}
    {
    }

    inline ~Octave() = default;

    inline auto width() const -> int
    {
      return _array.width();
    }

    inline auto height() const -> int
    {
      return _array.height();
    }

    inline auto scale_count() const -> int
    {
      return _array.depth();
    }

    inline auto init_texture(bool linear_interp = false) -> void
    {
      // Specify texture object parameters
      auto texture_descriptor = cudaTextureDesc{};
      memset(&texture_descriptor, 0, sizeof(texture_descriptor));
      texture_descriptor.addressMode[0] = cudaAddressModeClamp;
      texture_descriptor.addressMode[1] = cudaAddressModeClamp;
      texture_descriptor.addressMode[2] = cudaAddressModeClamp;
      texture_descriptor.filterMode =
          linear_interp ? cudaFilterModeLinear : cudaFilterModePoint;
      texture_descriptor.readMode = cudaReadModeElementType;
      texture_descriptor.normalizedCoords = false;

      _texture = _array.create_texture_object(texture_descriptor);
    }

    inline auto init_surface() -> const SurfaceObject&
    {
      _surface = _array.create_surface_object();
      return _surface;
    }

    inline auto deinit_texture() -> void
    {
      _texture.reset();
    }

    inline auto deinit_surface() -> void
    {
      _surface.~SurfaceObject();
    }

    inline auto array() -> ArrayView<T, 3>&
    {
      return _array;
    }

    inline auto array() const -> const ArrayView<T, 3>&
    {
      return _array;
    }

    inline auto texture_object() const noexcept
        -> const std::optional<TextureObject>&
    {
      return _texture;
    }

    inline auto surface_object() const noexcept -> const SurfaceObject&
    {
      return _surface;
    }

  private:
    Array<T, 3> _array;

    std::optional<TextureObject> _texture;
    SurfaceObject _surface;
  };


  template <typename T>
  inline auto make_gaussian_octave(int width, int height, int scale_count = 3)
  {
    return Octave<T>{width, height, scale_count + 3};
  }

  template <typename T>
  inline auto make_DoG_octave(int width, int height, int scale_count = 3)
  {
    return Octave<T>{width, height, scale_count + 2};
  }

  inline auto make_extremum_octave(int width, int height, int scale_count = 3)
  {
    return Octave<std::uint8_t>{width, height, scale_count};
  }

  template <typename T>
  inline auto make_gradient_octave(int width, int height, int scale_count = 3)
  {
    return Octave<Matrix<T, 2, 1>>{width, height, scale_count};
  }

}  // namespace DO::Shakti::Cuda
