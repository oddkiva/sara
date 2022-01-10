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

#ifndef DO_SHAKTI_MULTIARRAY_TEXTUREARRAY_HPP
#define DO_SHAKTI_MULTIARRAY_TEXTUREARRAY_HPP

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <DO/Shakti/Cuda/MultiArray/Matrix.hpp>
#include <DO/Shakti/Cuda/MultiArray/MultiArray.hpp>


namespace DO::Shakti {

  template <typename T>
  struct ChannelFormatDescriptor
  {
    static inline cudaChannelFormatDesc type()
    {
      return cudaCreateChannelDesc<T>();
    }
  };

  template <>
  struct ChannelFormatDescriptor<half>
  {
    static inline cudaChannelFormatDesc type()
    {
      return cudaCreateChannelDescHalf();
    }
  };

  template <>
  struct ChannelFormatDescriptor<Vector2f>
  {
    static inline cudaChannelFormatDesc type()
    {
      cudaChannelFormatDesc format = {32, 32, 0, 0, cudaChannelFormatKindFloat};
      return format;
    }
  };

  template <>
  struct ChannelFormatDescriptor<Vector4f>
  {
    static inline cudaChannelFormatDesc type()
    {
      cudaChannelFormatDesc format = {32, 32, 32, 32,
                                      cudaChannelFormatKindFloat};
      return format;
    }
  };

  template <>
  struct ChannelFormatDescriptor<Matrix2f>
  {
    static inline cudaChannelFormatDesc type()
    {
      cudaChannelFormatDesc format = {32, 32, 32, 32,
                                      cudaChannelFormatKindFloat};
      return format;
    }
  };

  //! \brief Wrapper class of the `cudaArray` object.
  template <typename T>
  class TextureArray
  {
    using self_type = TextureArray;

  public:
    inline TextureArray() = default;

    inline TextureArray(const Vector2i& sizes)
      : _sizes{sizes}
    {
      auto channel_descriptor = ChannelFormatDescriptor<T>::type();
      SHAKTI_SAFE_CUDA_CALL(
          cudaMallocArray(&_array, &channel_descriptor, sizes(0), sizes(1)));
    }

    inline TextureArray(const T* data, const Vector2i& sizes,
                        cudaMemcpyKind kind = cudaMemcpyHostToDevice)
      : self_type{sizes}
    {
      copy_from(data, sizes, sizes[0] * sizeof(T), kind);
    }

    inline TextureArray(const T* data, const Vector2i& sizes, size_t pitch,
                        cudaMemcpyKind kind = cudaMemcpyHostToDevice)
      : self_type{sizes}
    {
      copy_from(data, sizes, pitch, kind);
    }

    inline ~TextureArray()
    {
      const auto ret = cudaFreeArray(_array);
      if (ret != cudaSuccess)
        SHAKTI_STDERR << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
    }

    inline void copy_from(const T* data, const Vector2i& sizes, size_t pitch,
                          cudaMemcpyKind kind)
    {
      SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2DToArray(
          _array, 0, 0, data, pitch, sizes[0] * sizeof(T), sizes[1], kind));
    }

    inline void copy_to_host(T* host_data)
    {
      SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2DFromArray(
          host_data,                  // destination pointer
          _sizes[0] * sizeof(T),      // destination pitch
          _array,                     // source
          0, 0,                       // source (x, y)-offset
          _sizes[0] * sizeof(float),  // source width in bytes
          _sizes[1],                  // source height
          cudaMemcpyDeviceToHost));
    }

    inline void copy_to_device(MultiArray<T, 2>& device_array)
    {
      SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2DFromArray(
          device_array.data(), device_array.pitch(),  // destination
          _array,                                     // source array
          0, 0,                                       // source (x, y)-offset
          _sizes[0] * sizeof(float),                  // source width in bytes
          _sizes[1],                                  // source height
          cudaMemcpyDeviceToDevice));
    }

    inline operator cudaArray*() const
    {
      return _array;
    }

    inline const Vector2i& sizes() const
    {
      return _sizes;
    }

  protected:
    cudaArray* _array{nullptr};
    Vector2i _sizes{Vector2i::Zero()};
  };

}  // namespace DO::Shakti


#endif /* DO_SHAKTI_MULTIARRAY_TEXTUREARRAY_HPP */
