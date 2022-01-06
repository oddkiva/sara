// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <cuda_runtime.h>

#include <DO/Shakti/Cuda/MultiArray/TextureArray.hpp>


namespace DO::Shakti::Cuda {

  template <typename T>
  class Octave
  {
  public:
    inline Octave() = default;

    inline Octave(int width, int height, int scale_count)
      : _sizes{width, height, scale_count}
    {
      auto channel_descriptor = ChannelFormatDescriptor<T>::type();
      SHAKTI_SAFE_CUDA_CALL(cudaMalloc3DArray(
          &_array,              //
          &channel_descriptor,  //
          cudaExtent{
              .width = static_cast<unsigned long>(width),       //
              .height = static_cast<unsigned long>(height),     //
              .depth = static_cast<unsigned long>(scale_count)  //
          },
          cudaArrayLayered | cudaArraySurfaceLoadStore));
    }

    inline ~Octave()
    {
      deinit_layered_texture();
      deinit_surface();
      SHAKTI_SAFE_CUDA_CALL(cudaFreeArray(_array));
    }

    inline auto width() noexcept -> auto&
    {
      return _sizes(0);
    }

    inline auto height() noexcept -> auto&
    {
      return _sizes(1);
    }

    inline auto scale_count() noexcept -> auto&
    {
      return _sizes(2);
    }

    inline auto width() const noexcept -> const auto&
    {
      return _sizes(0);
    }

    inline auto height() const noexcept -> const auto&
    {
      return _sizes(1);
    }

    inline auto scale_count() const noexcept -> const auto&
    {
      return _sizes(2);
    }

    inline auto init_layered_texture(bool linear_interp = false) -> void
    {
      // Specify texture
      auto resource_descriptor = cudaResourceDesc{};
      memset(&resource_descriptor, 0, sizeof(resource_descriptor));
      resource_descriptor.resType = cudaResourceTypeArray;
      resource_descriptor.res.array.array = _array;

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

      cudaCreateTextureObject(&_texture_object, &resource_descriptor,
                              &texture_descriptor, nullptr);
    }

    inline auto init_surface() -> void
    {
      auto resource_descriptor = cudaResourceDesc{};
      memset(&resource_descriptor, 0, sizeof(resource_descriptor));
      resource_descriptor.resType = cudaResourceTypeArray;
      resource_descriptor.res.array.array = _array;

      // Create the surface objects
      cudaCreateSurfaceObject(&_surface_object, &resource_descriptor);
    }

    inline auto deinit_layered_texture() -> void
    {
      if (_texture_object != 0)
      {
        SHAKTI_SAFE_CUDA_CALL(cudaDestroyTextureObject(_texture_object));
        _texture_object = 0;
      }
    }

    inline auto deinit_surface() -> void
    {
      if (_surface_object != 0)
      {
        SHAKTI_SAFE_CUDA_CALL(cudaDestroySurfaceObject(_surface_object));
        _surface_object = 0;
      }
    }

    inline operator cudaArray_t() const
    {
      return _array;
    }

    inline auto texture_object() const noexcept
    {
      return _texture_object;
    }

    inline auto surface_object() const noexcept -> const auto&
    {
      return _surface_object;
    }

  private:
    cudaArray_t _array = nullptr;
    Vector3i _sizes = Vector3i::Zero();

    cudaTextureObject_t _texture_object = 0ull;
    cudaSurfaceObject_t _surface_object = 0ull;
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
