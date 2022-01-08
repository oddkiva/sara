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

#include <DO/Shakti/Cuda/MultiArray/MultiArrayView.hpp>

#include "DO/Sara/Core/Meta.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>


namespace DO { namespace Shakti {

  //! @brief Device multi-array class.
  template <typename T, int N, typename Strides = RowMajorStrides>
  class MultiArray : public MultiArrayView<T, N, Strides>
  {
    using self_type = MultiArray;
    using base_type = MultiArrayView<T, N, Strides>;
    using strides_type = Strides;

    using base_type::_data;
    using base_type::_pitch;
    using base_type::_sizes;
    using base_type::_strides;

  public:
    using vector_type = typename base_type::vector_type;
    using slice_vector_type = typename base_type::slice_vector_type;

    using slice_type = typename base_type::slice_type;
    using const_slice_type = const MultiArray<T, N - 1, Strides>;

    using size_type = typename base_type::size_type;

  public:
    //! @{
    //! @brief Constructor.
    inline MultiArray() = default;

    inline MultiArray(const vector_type& sizes)
      : base_type{}
    {
      resize(sizes);
    }

    inline MultiArray(size_type size)
      : self_type{vector_type{int(size)}}
    {
      static_assert(N == 1, "MultiArray must be 1D");
    }

    inline MultiArray(const self_type& other)
      : self_type{other.sizes()}
    {
      if (N == 1)
        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy(_data, other._data,
                                         base_type::size() * sizeof(T),
                                         cudaMemcpyDeviceToDevice));
      else if (N == 2)
        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2D(
            (void*) _data, _pitch, (void*) other._data, other._pitch,
            _sizes[0] * sizeof(T), _sizes[1], cudaMemcpyDeviceToDevice));
      else if (N == 3)
      {
        auto params = cudaMemcpy3DParms{};
        params.srcPtr.ptr = (void*) other._data;
        params.srcPtr.pitch = other._pitch;
        params.srcPtr.xsize = other._sizes[0];
        params.srcPtr.ysize = other._sizes[1];

        params.dstPtr.ptr = reinterpret_cast<void*>(_data);
        params.dstPtr.pitch = _pitch;
        params.dstPtr.xsize = _sizes[0];
        params.dstPtr.ysize = _sizes[1];

        params.kind = cudaMemcpyDeviceToDevice;
        params.extent = make_cudaExtent(_sizes[0], _sizes[1], _sizes[2]);

        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy3D(&params));
      }
      else
        throw std::runtime_error{"Unsupported dimension!"};
    }

    inline MultiArray(const T* host_data, const vector_type& sizes)
      : self_type{sizes}
    {
      if (N == 1)
        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy(_data, host_data,
                                         base_type::size() * sizeof(T),
                                         cudaMemcpyHostToDevice));
      else if (N == 2)
        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2D(
            _data, _pitch, host_data, sizes[0] * sizeof(T),
            sizes[0] * sizeof(T), sizes[1], cudaMemcpyHostToDevice));
      else if (N == 3)
      {
        auto params = cudaMemcpy3DParms{};

        params.srcPtr =
            make_cudaPitchedPtr(const_cast<T*>(host_data), sizes[0] * sizeof(T),
                                sizes[0], sizes[1]);
        params.dstPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(_data),
                                            _pitch, sizes[0], sizes[1]);
        params.extent =
            make_cudaExtent(_sizes[0] * sizeof(T), _sizes[1], _sizes[2]);
        params.kind = cudaMemcpyHostToDevice;

        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy3D(&params));
      }
      else
        throw std::runtime_error{"Unsupported dimension!"};
    }

    inline MultiArray(self_type&& other)
      : self_type{}
    {
      std::swap(_data, other._data);
      _sizes = other._sizes;
      _strides = other._strides;
      _pitch = other._pitch;
    }
    //! @}

    //! @brief Assignment operator uses the copy-swap idiom.
    self_type& operator=(self_type other)
    {
      swap(other);
      return *this;
    }

    //! @brief Destructor.
    inline ~MultiArray()
    {
      SHAKTI_SAFE_CUDA_CALL(cudaFree(_data));
    }

    //! @brief Resize the multi-array.
    inline void resize(const vector_type& sizes)
    {
      if (_sizes == sizes)
        return;

      SHAKTI_SAFE_CUDA_CALL(cudaFree(_data));

      _sizes = sizes;
      _strides = strides_type::compute(sizes);

      auto void_data = reinterpret_cast<void**>(&_data);

      if (N == 1)
      {
        const auto byte_size = sizeof(T) * this->base_type::size();
        SHAKTI_SAFE_CUDA_CALL(cudaMalloc(void_data, byte_size));
      }
      else if (N == 2)
      {
        SHAKTI_SAFE_CUDA_CALL(cudaMallocPitch(
            void_data, &_pitch, _sizes[0] * sizeof(T), _sizes[1]));
      }
      else if (N == 3)
      {
        cudaPitchedPtr pitched_device_ptr = {0, 0, 0, 0};
        auto extent = make_cudaExtent(sizes[0] * sizeof(T), sizes[1], sizes[2]);
        SHAKTI_SAFE_CUDA_CALL(cudaMalloc3D(&pitched_device_ptr, extent));
        _pitch = pitched_device_ptr.pitch;
        *void_data = pitched_device_ptr.ptr;
      }
      else
        throw std::runtime_error{"Unsupported dimension!"};
    }

    //! @brief Swap multi-array objects.
    void swap(self_type& other)
    {
      using std::swap;
      swap(_sizes, other._sizes);
      swap(_strides, other._strides);
      swap(_pitch, other._pitch);
    }
  };

  template <typename T>
  using Array = MultiArray<T, 1>;

}}  // namespace DO::Shakti
