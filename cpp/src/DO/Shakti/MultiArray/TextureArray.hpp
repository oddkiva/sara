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

#include <cuda_runtime.h>

#include <DO/Shakti/MultiArray/Matrix.hpp>


namespace DO { namespace Shakti {

  template <typename T>
  struct ChannelFormatDescriptor
  {
    static inline cudaChannelFormatDesc type()
    {
      return cudaCreateChannelDesc<T>();
    }
  };

  template <>
  struct ChannelFormatDescriptor<Vector2f>
  {
    static inline cudaChannelFormatDesc type()
    {
      cudaChannelFormatDesc format = {
        32, 32, 0, 0, cudaChannelFormatKindFloat
      };
      return format;
    }
  };

  template <>
  struct ChannelFormatDescriptor<Vector4f>
  {
    static inline cudaChannelFormatDesc type()
    {
      cudaChannelFormatDesc format = {
        32, 32, 32, 32, cudaChannelFormatKindFloat
      };
      return format;
    }
  };

  template <>
  struct ChannelFormatDescriptor<Matrix2f>
  {
    static inline cudaChannelFormatDesc type()
    {
      cudaChannelFormatDesc format = {
        32, 32, 32, 32, cudaChannelFormatKindFloat
      };
      return format;
    }
  };

  //! \brief Wrapper class of the `cudaArray` object.
  template <typename T>
  class TextureArray
  {
    using self_type = TextureArray;

  public:
    __host__
    TextureArray() = default;

    __host__
    inline TextureArray(const Vector2i& sizes)
      : _sizes{ sizes }
    {
      auto channel_descriptor = ChannelFormatDescriptor<T>::type();
      SHAKTI_SAFE_CUDA_CALL(cudaMallocArray(
        &_array, &channel_descriptor, sizes(0), sizes(1)));
    }

    __host__
    inline TextureArray(const T *data, const Vector2i& sizes,
                        cudaMemcpyKind kind = cudaMemcpyHostToDevice)
      : self_type{ sizes }
    {
      copy_from(data, sizes, sizes[0] * sizeof(T), kind);
    }

    __host__
    inline TextureArray(const T *data, const Vector2i& sizes, size_t pitch,
                        cudaMemcpyKind kind = cudaMemcpyHostToDevice)
      : self_type{ sizes }
    {
      copy_from(data, sizes, pitch, kind);
    }

    __host__
    inline ~TextureArray()
    {
      SHAKTI_SAFE_CUDA_CALL(cudaFreeArray(_array));
    }

    __host__
    inline void copy_from(const T *data, const Vector2i& sizes, size_t pitch,
                          cudaMemcpyKind kind)
    {
      SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2DToArray(
        _array, 0, 0, data, pitch, sizes[0] * sizeof(T), sizes[1], kind));
    }

    __host__
    inline void copy_to_host(T *host_data)
    {
      SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2DFromArray(
        host_data, _sizes[0] * sizeof(T), _array,
        0, 0, _sizes[0]*sizeof(float), _sizes[1], cudaMemcpyDeviceToHost));
    }

    __host__
    inline operator cudaArray *() const
    {
      return _array;
    }

    __host__
    inline const Vector2i& sizes() const
    {
      return _sizes;
    }

  protected:
    cudaArray *_array{ nullptr };
    Vector2i _sizes{ Vector2i::Zero() };
  };

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_MULTIARRAY_TEXTUREARRAY_HPP */