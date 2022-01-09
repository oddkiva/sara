#pragma once

#include <DO/Sara/Core/Image/Image.hpp>

#include <DO/Shakti/Cuda/MultiArray/Matrix.hpp>
#include <DO/Shakti/Cuda/MultiArray/SurfaceObject.hpp>
#include <DO/Shakti/Cuda/MultiArray/TextureObject.hpp>

#include <cuda_runtime.h>

#include <optional>


namespace DO::Shakti::Cuda {

  //! @brief Wrapper class of the cudaArray_t object.
  template <typename T, int N>
  class ArrayView
  {
    inline ArrayView() noexcept = default;

    inline ArrayView(cudaArray_t arr, const Vector<int, N>& sizes,
                     unsigned flags = 0) noexcept
      : _array{arr}
      , _sizes{sizes}
      , _flags{flags}
    {
    }

    inline ArrayView(const ArrayView&) noexcept = default;

    inline ArrayView(ArrayView&&) noexcept = default;

    inline ~ArrayView() noexcept = default;

    inline auto operator=(const ArrayView&) const noexcept -> auto& = default;

    inline operator cudaArray_t() const noexcept
    {
      return _array;
    }

    inline auto crop(const Vector<int, N>& begin,
                     const Vector<int, N>& end) const noexcept -> const auto&
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

    inline auto begin() const noexcept -> auto
    {
      return _begin.has_value()
        : *_begin : Vector<int, N>::Zero();
    }

    inline auto end() const noexcept -> auto
    {
      return _end.has_value()
        : *_end : _sizes;
    }

    inline auto cropped_sizes() const noexcept -> auto
    {
      return _begin.has_value() && _end.has_value() ? *_end - *_begin : _sizes;
    }

    inline auto sizes() const noexcept -> const auto&
    {
      return _sizes;
    }

    inline auto copy_from(const ArrayView& other) -> void
    {
      static_assert(N == 2, "Not implemented!");
      if (sizes() != other.sizes())
        throw std::runtime_error{"Cannot copy array of different sizes"};
      SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2DArrayToArray(
          _array,  //
          begin()->x() : 0,
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

    inline auto create_texture_object(const cudaTextureDesc& desc) const
        -> TextureObject
    {
      return {_array, desc};
    }

    inline auto create_surface_object() const -> SurfaceObject
    {
      if (!(flags & cudaArraySurfaceLoadStore))
        throw std::runtime_error{"Error: cannot create surface object. The "
                                 "array must be created with the "
                                 "cudaArraySurfaceLoadStore flag!"};
      return {_array};
    }

    static inline auto channel_descriptor()
    {
      return ChannelFormatDescriptor<T>::type();
    }

  protected:
    cudaArray_t _array = nullptr;
    unsigned int _flags = 0;
    Vector<int, N> _sizes = Vector<int, N>::Zero();
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
      if constexpr (N == 2)
      {
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

      else if constexpr (N == 3)
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
            flags));
      }
      else
      {
        throw std::runtime_error{"Not Implemented!"};
      }
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

    inline Array(Array&& other) noexcept
      : base_type{other}
      , _flags{other.flags}
    {
    }

    inline auto operator=(const Array& other) const -> auto&
    {
      copy_from(other);
      return *this;
    }
  };

}  // namespace DO::Shakti::Cuda
