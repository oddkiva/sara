#pragma once

#include <DO/Sara/Core/Image/Image.hpp>

#include <DO/Shakti/Cuda/Utilities/ErrorCheck.hpp>

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
  public:
    inline ArrayView() = default;

    inline ArrayView(cudaArray_t arr, const Vector<int, N>& sizes,
                     unsigned flags = 0)
      : _array{arr}
      , _sizes{sizes}
      , _flags{flags}
    {
    }

    inline ArrayView(const ArrayView&) = default;

    inline ArrayView(ArrayView&&) noexcept = default;

    inline ~ArrayView() = default;

    inline auto operator=(const ArrayView&) -> ArrayView& = default;

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

    inline auto begin() const noexcept -> Vector<int, N>
    {
      return _begin.has_value() ? *_begin : Vector<int, N>::Zero();
    }

    inline auto end() const noexcept -> Vector<int, N>
    {
      return _end.has_value() ? *_end : _sizes;
    }

    inline auto cropped_sizes() const noexcept -> Vector<int, N>
    {
      return _begin.has_value() && _end.has_value() ? *_end - *_begin : _sizes;
    }

    inline auto width() const noexcept -> int
    {
      return cropped_sizes()(0);
    }

    inline auto height() const -> int
    {
      return cropped_sizes()(1);
    }

    inline auto depth() const -> int
    {
      return cropped_sizes()(2);
    }

    inline auto sizes() const noexcept -> const Vector<int, N>&
    {
      return _sizes;
    }

    inline auto swap(ArrayView& other) -> void
    {
      std::swap(_array, other._array);
      std::swap(_flags, other._flags);
      std::swap(_sizes, other._sizes);
      std::swap(_begin, other._begin);
      std::swap(_end, other._end);
    }

    inline auto copy_from(const ArrayView& other) -> void
    {
      static_assert(N == 2, "Not implemented!");
      if (sizes() != other.sizes())
        throw std::runtime_error{"Cannot copy array of different sizes"};
      SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2DArrayToArray(_array,        //
                                                     begin()(0),    //
                                                     begin()(1),    //
                                                     other._array,  //
                                                     other.begin()(0),
                                                     other.begin()(1),  //
                                                     _sizes.x(), _sizes.y()));
    }

    inline auto copy_from(const Sara::ImageView<T, N>& host_data) -> void
    {
      const auto b = begin();
      const auto sizes = cropped_sizes();

      if constexpr (N == 2)
      {
        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2DToArray(
            _array, b(0), b(1), host_data.data(), host_data.width(), sizes(0),
            sizes(1), cudaMemcpyHostToDevice));
      }
      else if constexpr (N == 3)
      {
        if (host_data.width() != sizes(0) ||   //
            host_data.height() != sizes(1) ||  //
            host_data.depth() != sizes(2))
          throw std::runtime_error{"Invalid sizes!"};

        auto copy_params = cudaMemcpy3DParms{};
        {
          copy_params.srcPtr = make_cudaPitchedPtr(                       //
              reinterpret_cast<void*>(const_cast<T*>(host_data.data())),  //
              host_data.width() * sizeof(T), host_data.width(),
              host_data.height()  //
          );
          copy_params.srcPos = make_cudaPos(0, 0, 0);

          copy_params.dstArray = _array;
          copy_params.dstPos = make_cudaPos(b(0), b(1), b(2));

          // Because we use a CUDA array the extent is in terms of number of
          // elements and not in bytes.
          copy_params.extent = make_cudaExtent(
              host_data.width(), host_data.height(), host_data.depth());
          copy_params.kind = cudaMemcpyHostToDevice;
        }

        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy3D(&copy_params));
      }
      else
        throw std::runtime_error{
            "Unsupported copy operation for the requested dimensions!"};
    }

    inline auto copy_to(Sara::ImageView<T, N>& host_data) const -> void
    {
      const auto beg = begin();
      const auto sizes = cropped_sizes();

      if constexpr (N == 3)
      {
        if (sizes(0) != host_data.width() ||   //
            sizes(1) != host_data.height() ||  //
            sizes(2) != host_data.depth())
          throw std::runtime_error{"Invalid sizes!"};

        auto copy_params = cudaMemcpy3DParms{};
        {
          copy_params.srcArray = _array;
          copy_params.srcPos = make_cudaPos(beg(0), beg(1), beg(2));
          copy_params.dstPtr = make_cudaPitchedPtr(       //
              reinterpret_cast<void*>(host_data.data()),  //
              host_data.width() * sizeof(T), host_data.width(),
              host_data.height()  //
          );
          copy_params.dstPos = make_cudaPos(0, 0, 0);

          // Because we use a CUDA array the extent is in terms of number of
          // elements and not in bytes.
          copy_params.extent = make_cudaExtent(sizes(0), sizes(1), sizes(2));
          copy_params.kind = cudaMemcpyDeviceToHost;
        }

        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy3D(&copy_params));
      }
      else
        throw std::runtime_error{
            "Unsupported copy operation for this dimension!"};
    }

    inline auto create_texture_object(const cudaTextureDesc& desc) const
        -> TextureObject
    {
      return {_array, desc};
    }

    inline auto create_surface_object() const -> SurfaceObject
    {
      if (!(_flags & cudaArraySurfaceLoadStore))
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
    using base_type::_begin;
    using base_type::_end;
    using base_type::_flags;
    using base_type::_sizes;

  public:
    inline Array() = default;

    inline Array(const Shakti::Vector<int, N>& sizes, unsigned int flags = 0)
      : base_type{nullptr, sizes, flags}
    {
      auto channel_descriptor = ChannelFormatDescriptor<T>::type();

      if constexpr (N == 2)
      {
        SHAKTI_SAFE_CUDA_CALL(
            cudaMallocArray(&_array,              //
                            &channel_descriptor,  //
                            cudaExtent{
                                static_cast<unsigned long>(sizes(0)),  //
                                static_cast<unsigned long>(sizes(1))   //
                            },
                            _flags));
      }

      else if constexpr (N == 3)
      {
        SHAKTI_SAFE_CUDA_CALL(
            cudaMalloc3DArray(&_array,              //
                              &channel_descriptor,  //
                              cudaExtent{
                                  static_cast<unsigned long>(sizes(0)),  //
                                  static_cast<unsigned long>(sizes(1)),  //
                                  static_cast<unsigned long>(sizes(2))   //
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
      SHAKTI_SAFE_CUDA_CALL(
          cudaMallocArray(&_array,              //
                          &channel_descriptor,  //
                          cudaExtent{
                              static_cast<unsigned long>(_sizes(0)),  //
                              static_cast<unsigned long>(_sizes(1)),  //
                          },
                          _flags));
      copy_from(other);
    }

    inline Array(Array&& other) noexcept
    {
      swap(other);
    }

    ~Array()
    {
      const auto ret = cudaFreeArray(_array);
      if (ret != cudaSuccess)
        SHAKTI_STDERR << cudaGetErrorString(cudaDeviceSynchronize())
                      << std::endl;
    }

    inline auto operator=(const Array& other) const -> auto&
    {
      copy_from(other);
      return *this;
    }
  };

  template <typename T>
  inline auto make_2d_array(const Vector2i& sizes) -> Array<T, 2>
  {
    return {sizes, 0};
  }

  template <typename T>
  inline auto make_3d_array(const Vector3i& sizes) -> Array<T, 3>
  {
    return {sizes, 0};
  }

  template <typename T>
  inline auto make_3d_layered_array(const Vector3i& sizes) -> Array<T, 3>
  {
    return {sizes, cudaArrayLayered};
  }

  template <typename T>
  inline auto make_3d_surface_array(const Vector3i& sizes) -> Array<T, 3>
  {
    return {sizes, cudaArraySurfaceLoadStore};
  }

  template <typename T>
  inline auto make_3d_layered_surface_array(const Vector3i& sizes)
      -> Array<T, 3>
  {
    return {sizes, cudaArrayLayered | cudaArraySurfaceLoadStore};
  }

}  // namespace DO::Shakti::Cuda
