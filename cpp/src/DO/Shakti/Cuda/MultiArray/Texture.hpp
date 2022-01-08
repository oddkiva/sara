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

#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Shakti/Cuda/MultiArray/TextureArray.hpp>

#include <optional>


namespace DO::Shakti::Cuda {

  //! @brief Wrapper class of the cudaArray_t object.
  template <typename T, int N>
  class ArrayView
  {
    inline ArrayView() noexcept = default;

    inline ArrayView(cudaArray_t arr, int width, int height)
      : _array{arr}
      , _sizes{width, height}
    {
    }

    inline ArrayView(const ArrayView&) noexcept = default;

    inline ArrayView(ArrayView&&) noexcept = default;

    inline ~ArrayView() = default;

    inline auto operator=(const ArrayView&) const -> auto& = default;

    inline operator cudaArray_t() const
    {
      return _array;
    }

    inline auto crop(const Vector2i& begin, const Vector2i& end) const noexcept
        -> const auto&
    {
      _begin = begin;
      _end = end;
      return *this;
    }

    inline auto reset_cropped_sizes() noexcept -> void
    {
      _begin.reset();
      _end.reset();
    }

    inline auto cropped_sizes() const noexcept -> const auto&
    {
      return _begin.has_value() && _end.has_value() ? *_end - *_begin : _sizes;
    }

    inline auto begin() const noexcept -> Vector<int, N>
    {
      return _begin.has_value() : *_begin : Vector<int, N>::Zero();
    }

    inline auto end() const noexcept -> Vector<int, N>
    {
      return _end.has_value() : *_end : _sizes;
    }

    inline auto sizes() const noexcept -> const auto&
    {
      return _sizes;
    }

    inline auto copy_from(const ArrayView& other) -> void
    {
      if (sizes() != other.sizes())
        throw std::runtime_error{"Cannot copy array of different sizes"};
      SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2DArrayToArray(
          _array,  //
          _begin.has_value() ? _begin->x() : 0,
          _begin.has_value() ? _begin->y() : 0,  //
          other._array,                          //
          other._begin.has_value() ? other._begin->x() : 0,
          other._begin.has_value() ? other._begin->y() : 0,  //
          _sizes.x(), _sizes.y()));
      return *this;
    }

    inline auto copy_from(const Sara::ImageView<T, N>& host_data) -> void
    {
      static_assert(N == 2, "Not Implemented!");
      SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2DToArray(
          _array, begin(0), begin(1), host_data.data(), host_data.width(),
          host_data.width(), host_data.height(), cudaMemcpyHostToDevice));
    }

  protected:
    cudaArray_t _array = nullptr;
    Vector<int, N> _sizes;
    std::optional<Vector<int, N>> _begin;
    std::optional<Vector<int, N>> _end;
  };

  //! @brief Wrapper class of the cudaArray_t object.
  template <typename T, int N>
  class Array : public ArrayView<T, N>
  {
    using base_type = ArrayView<T, N>;
    using base_type::_array;
    using base_type::_sizes;

  public:
    inline Array() noexcept = default;

    inline Array(const Vector<int, N>& sizes, unsigned int flags = 0)
      : base_type::_sizes{sizes}
      , _flags{flags}
    {
      static_assert(N == 2, "Not implemented");

      _channel_descriptor = ChannelFormatDescriptor<T>::type();
      SHAKTI_SAFE_CUDA_CALL(cudaMallocArray(
          &_array,              //
          &channel_descriptor,  //
          cudaExtent{
              .width = static_cast<unsigned long>(sizes(0)),   //
              .height = static_cast<unsigned long>(sizes(1)),  //
          },
          _flags));
    }

    inline Array(const Array& other)
      : base_type{other._sizes, other._flags}
    {
      static_assert(N == 2, "Not implemented");

      auto channel_descriptor = ChannelFormatDescriptor<T>::type();
      SHAKTI_SAFE_CUDA_CALL(cudaMallocArray(
          &_array,              //
          &channel_descriptor,  //
          cudaExtent{
              .width = static_cast<unsigned long>(_sizes(0)),   //
              .height = static_cast<unsigned long>(_sizes(1)),  //
          },
          _flags));
      copy_from(other);
    }

    inline Array(Array&& other)
      : base_type{other}
      , _flags{other.flags}
    {
    }

    inline auto operator=(const Array& other) const -> auto&
    {
      copy_from(other);
      return *this;
    }

  private:
    unsigned int _flags = 0;
  };

  //! @brief Wrapper class of the Texture object.
  template <typename T, int N>
  class Texture
  {
    using self_type = Texture;

  public:
    inline Texture() = default;

    inline Texture(const ArrayView<T, N>& array, bool bilinear_interp = true,
                   bool normalized_coords = false)
      :_array{array}
    {
      static_assert(N == 2, "Texture must be 2D!");

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
      texture_descriptor.filterMode =
          bilinear_interp ? cudaFilterModeLinear : cudaFilterModePoint;
      texture_descriptor.readMode = cudaReadModeElementType;
      texture_descriptor.normalizedCoords = normalized_coords;

      cudaCreateTextureObject(&_texture_object, &resource_descriptor,
                              &texture_descriptor, nullptr);
    }

    inline ~Texture()
    {
      if (_texture_object != 0)
      {
        SHAKTI_SAFE_CUDA_CALL(cudaDestroyTextureObject(_texture_object));
        _texture_object = 0;
      }
    }

    inline auto array() const -> const ArrayView<T, N>&
    {
      return _array;
    }

    inline auto texture_object() const noexcept
    {
      return _texture_object;
    }


  protected:
    ArrayView<T, N> _array;
    cudaTextureObject_t _texture_object = 0;
  };

}  // namespace DO::Shakti::Cuda
