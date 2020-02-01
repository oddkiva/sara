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

#ifndef DO_SHAKTI_MULTIARRAY_MULTIARRAYVIEW_HPP
#define DO_SHAKTI_MULTIARRAY_MULTIARRAYVIEW_HPP

#include <DO/Shakti/Utilities/ErrorCheck.hpp>

#include <DO/Shakti/MultiArray/Matrix.hpp>
#include <DO/Shakti/MultiArray/Strides.hpp>


namespace DO { namespace Shakti {

  //! @brief ND-array view class.
  template <typename T, int N, typename Strides>
  class MultiArrayView
  {
    static_assert(N > 0, "Dimension N must be positive");

    using self_type = MultiArrayView;
    using strides_type = Strides;

  public: /* typedefs. */
    //! Storage order.
    enum { Dimension = N };

    //! @{
    //! @brief STL-compatible interface.
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T *;
    using const_iterator = const T *;
    //! @}

    //! @{
    //! @brief Slice type.
    using slice_type = MultiArrayView<T, N-1, Strides>;
    using const_slice_type = const MultiArrayView<T, N-1, Strides>;
    //! @}

    //! @{
    //! @brief Vector type.
    using vector_type = Vector<int, N>;
    using slice_vector_type = Vector<int, N-1>;
    //! @}

  public: /* methods */
    //! @brief Default constructor.
    __host__ __device__
    inline MultiArrayView() = default;

    //! @{
    //! @brief Constructor that wraps plain data with its known sizes.
    __host__ __device__
    inline MultiArrayView(value_type *data, const vector_type& sizes)
      : _data{ data }
      , _sizes{ sizes }
      , _strides{ strides_type::compute(sizes) }
      , _pitch{ sizes[0] * sizeof(T) }
    {
    }

    __host__ __device__
    inline MultiArrayView(value_type *data, const vector_type& sizes,
                          size_type pitch)
      : _data{ data }
      , _sizes{ sizes }
      , _strides{ strides_type::compute(sizes) }
      , _pitch{ pitch }
    {
      if (_sizes[0] * sizeof(T) > pitch)
        throw std::runtime_error{ "_sizes[0]*sizeof(T) > pitch" };
    }
    //! @}

    //! @brief Return the size vector of the MultiArray object.
    __host__ __device__
    inline const vector_type& sizes() const
    {
      return _sizes;
    }

    //! @brief Return the pitch size in bytes of the MultiArray object.
    __host__ __device__
    inline size_type pitch() const
    {
      return _pitch;
    }

    //! @brief Return the padded width of the MultiArray object.
    //! This is useful when computing the grid.
    __host__ __device__
    inline int padded_width() const
    {
      return int(pitch() / sizeof(T));
    }

    //! @brief Return the number of elements in the internal data array.
    __host__ __device__
    inline size_type size() const
    {
      return compute_size(_sizes);
    }

    //! @brief Return the i-th sizes of the multi-array.
    __host__ __device__
    inline int size(int i) const
    {
      return _sizes(i);
    }

    //! @brief Return the number of rows.
    __host__ __device__
    inline int width() const
    {
      return _sizes(0);
    }

    //! @brief Return the number of columns.
    __host__ __device__
    inline int height() const
    {
      return _sizes(1);
    }

    //! @brief Return the depth size.
    __host__ __device__
    inline int depth() const
    {
      return _sizes(2);
    }

    //! @brief Return the stride vector of the MultiArray object.
    __host__ __device__
    inline const vector_type& strides() const
    {
      return _strides;
    }

    //! @brief Return the stride value along the i-th dimension.
    __host__ __device__
    inline int stride(int i) const
    {
      return _strides(i);
    }

    //! @{
    //! @brief Return the array pointer.
    __host__ __device__
    inline pointer data()
    {
      return _data;
    }

    __host__ __device__
    inline const_pointer data() const
    {
      return _data;
    }
    //! @}

    //! @brief Return true/false whether the array is empty.
    __host__ __device__
    inline bool empty() const
    {
      return _data == nullptr;
    }

    //! @{
    //! @brief Return the reference to array element at given coordinates.
    __host__ __device__
    inline reference operator()(const vector_type& pos)
    {
      return _data[offset(pos)];
    }

    __host__ __device__
    inline reference operator()(int i, int j)
    {
      static_assert(N == 2, "MultiArray must be 2D.");
      return _data[offset(Vector2i(i, j))];
    }

    __host__ __device__
    inline reference operator()(int i, int j, int k)
    {
      static_assert(N == 3, "MultiArray must be 3D.");
      return _data[offset(Vector3i(i, j, k))];
    }

    __host__ __device__
    inline const_reference operator()(const vector_type& pos) const
    {
      return _data[offset(pos)];
    }

    __host__ __device__
    inline const_reference operator()(int i, int j) const
    {
      static_assert(N == 2, "MultiArray must be 2D.");
      return _data[offset(Vector2i(i, j))];
    }

    __host__ __device__
    inline const_reference operator()(int i, int j, int k) const
    {
      static_assert(N == 3, "MultiArray must be 3D.");
      return _data[offset(Vector3i(i, j, k))];
    }
    //! @}

    //! @brief Copy the ND-array device array to host array.
    //! You must allocate the array with the appropriate size.
    __host__
    inline void copy_to_host(T *host_data) const
    {
      if (N == 2)
      {
        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy2D(
          host_data, _sizes[0] * sizeof(T), _data, _pitch, _sizes[0] * sizeof(T), _sizes[1],
          cudaMemcpyDeviceToHost));
      }
      else if (N == 3)
      {
        cudaMemcpy3DParms params = { 0 };

        params.srcPtr = make_cudaPitchedPtr(reinterpret_cast<void *>(_data),
                                            _pitch, _sizes[0], _sizes[1]);
        params.dstPtr = make_cudaPitchedPtr(host_data, _sizes[0] * sizeof(T),
                                            _sizes[0], _sizes[1]);
        params.extent = make_cudaExtent(_sizes[0]*sizeof(T), _sizes[1],
                                        _sizes[2]);
        params.kind = cudaMemcpyDeviceToHost;

        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy3D(&params));
      }
      else
        SHAKTI_SAFE_CUDA_CALL(cudaMemcpy(host_data, _data, sizeof(T) * size(),
                                         cudaMemcpyDeviceToHost));
    }

    //! @brief Copy the ND-array content to a std::vector object.
    __host__
    inline void copy_to_host(std::vector<T>& host_vector) const
    {
      if (host_vector.size() != size())
        host_vector.resize(size());
      copy_to_host(host_vector.data());
    }

  protected:
    //! @brief Compute the raw size needed to allocate the internal data.
    __host__ __device__
    inline size_type compute_size(const vector_type& sizes) const
    {
      size_t sz = 1;
      for (int i = 0; i < N; ++i)
        sz *= sizes(i);
      return sz;
    }

    //! @brief Compute the 1D index of the corresponding coordinates.
    __host__ __device__
    inline int offset(const vector_type& pos) const
    {
      return pos.dot(_strides);
    }

  protected: /* data members. */
    //! @brief Internal storage array.
    value_type *_data{ nullptr };
    //! @brief Vector of size along each dimension.
    vector_type _sizes{ vector_type::Zero() };
    //! @brief Vector of stride for each dimension.
    vector_type _strides{ vector_type::Zero() };
    //! @brief Pitch size in number of bytes.
    size_type _pitch{ 0 };
  };

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_MULTIARRAY_MULTIARRAYVIEW_HPP */
