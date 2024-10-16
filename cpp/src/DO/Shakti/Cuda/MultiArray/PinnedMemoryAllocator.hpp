// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/Cuda/Utilities/ErrorCheck.hpp>

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <string>


namespace DO { namespace Shakti {

  //! @brief CUDA pinned memory allocator.
  //! @{

  template <typename T>
  class PinnedMemoryAllocator;

  template <>
  class PinnedMemoryAllocator<void>
  {
  public:
    using value_type = void;
    using pointer = void*;
    using const_pointer = const void*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind
    {
      using other = PinnedMemoryAllocator<U>;
    };
  };

  template <typename T>
  class PinnedMemoryAllocator
  {
  public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind
    {
      using other = PinnedMemoryAllocator<U>;
    };

    __host__ __device__ inline PinnedMemoryAllocator()
    {
    }

    template <typename U>
    __host__ __device__ inline PinnedMemoryAllocator(const PinnedMemoryAllocator<U>&)
    {
    }

    __host__ __device__ inline pointer address(reference r)
    {
      return &r;
    }

    __host__ __device__ inline const_pointer address(const_reference r)
    {
      return &r;
    }

    __host__ inline pointer allocate(size_type cnt, const_pointer = 0)
    {
      if (cnt > this->max_size())
        throw std::bad_alloc{};

      pointer result{nullptr};

      const auto ret = cudaMallocHost(reinterpret_cast<void**>(&result),
                                  cnt * sizeof(value_type));
      if (ret != cudaSuccess)
        throw std::bad_alloc{};

      return result;
    }

    __host__ inline void deallocate(pointer p, size_type)
    {
      const auto ret = cudaFreeHost(p);
      if (ret != cudaSuccess)
        SHAKTI_STDERR << cudaGetErrorString(cudaDeviceSynchronize())
                      << std::endl;
    }

    __host__ __device__ inline size_type max_size() const
    {
      return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    __host__ __device__ inline bool operator==(PinnedMemoryAllocator const&)
    {
      return true;
    }

    __host__ __device__ inline bool operator!=(PinnedMemoryAllocator const& x)
    {
      return !operator==(x);
    }
  };

  //! @}

}}  // namespace DO::Shakti
