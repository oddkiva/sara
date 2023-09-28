// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
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

  //! @brief Wrapper class of the Surface object.
  class SurfaceObject
  {
  public:
    inline SurfaceObject() = default;

    inline SurfaceObject(cudaArray_t array)
      : _array{array}
    {
      auto resource_descriptor = cudaResourceDesc{};
      memset(&resource_descriptor, 0, sizeof(resource_descriptor));
      resource_descriptor.resType = cudaResourceTypeArray;
      resource_descriptor.res.array.array = _array;

      // Create the surface objects
      SHAKTI_SAFE_CUDA_CALL(cudaCreateSurfaceObject(&_surface_object, &resource_descriptor));
      SHAKTI_STDOUT << "Attaching surface object = " << _surface_object << " to array " << _array << std::endl;
    }

    // The copy constructor is forbidden. Only one surface object per
    // cudaArray_t object!
    inline SurfaceObject(const SurfaceObject& other) = delete;

    //! @brief We can move surface object but not copy.
    inline SurfaceObject(SurfaceObject&& other) noexcept = default;

    inline ~SurfaceObject()
    {
      if (_surface_object != 0)
      {
        SHAKTI_STDOUT << "Destroying surface object" << std::endl;
        const auto ret = cudaDestroySurfaceObject(_surface_object);
        if (ret != cudaSuccess)
          SHAKTI_STDERR << cudaGetErrorString(cudaDeviceSynchronize())
                        << std::endl;
        _array = nullptr;
        _surface_object = 0;
      }
    }

    inline operator cudaSurfaceObject_t() const noexcept
    {
      return _surface_object;
    }

    inline auto initialized() const -> bool
    {
      return _array != nullptr && _surface_object != 0ull;
    }

    inline auto operator=(SurfaceObject other) -> SurfaceObject&
    {
      swap(other);
      return *this;
    }

    inline auto swap(SurfaceObject& other) -> void
    {
      std::swap(_array, other._array);
      std::swap(_surface_object, other._surface_object);
    }

  protected:
    cudaArray_t _array = nullptr;
    cudaSurfaceObject_t _surface_object = 0ull;
  };

}  // namespace DO::Shakti::Cuda
