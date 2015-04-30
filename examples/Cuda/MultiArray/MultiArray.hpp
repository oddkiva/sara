#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>

#include <Utilities/ErrorCheck.hpp>

#include "Matrix.hpp"


namespace DO { namespace Device {


  //! Only row-major ND-array for now.
  template <typename T, int N>
  class MultiArray
  {
  public:
    using self_type = MultiArray;
    using vector_type = Vector<int, N>;

  public:
    __host__
    inline MultiArray() = default;

    __host__
    inline MultiArray(const vector_type& sizes)
      : MultiArray()
    {
      resize(sizes);
    }

    __host__
    inline MultiArray(const self_type& other)
      : MultiArray()
    {
      resize(other._sizes);
      CHECK_CUDA_RUNTIME_ERROR(
        cudaMemcpy(_data, other._begin, other.size() * sizeof(T),
                   cudaMemcpyHostToDevice));
    }

    __host__
    inline ~MultiArray()
    {
      CHECK_CUDA_RUNTIME_ERROR(cudaFree(_data));
    }

    __host__ __device__
    inline vector_type sizes() const
    {
      return _sizes;
    }

    __host__ __device__
    inline size_t size() const
    {
      size_t sz{1};
      for (int i = 0; i < N; ++i)
        sz *= _sizes(i);
      return sz;
    }

    __host__ __device__
    inline bool empty() const
    {
      return _data == nullptr;
    }

    __host__ __device__
    inline T * data()
    {
      return _data;
    }

    __host__ __device__
    inline const T * data() const
    {
      return _data;
    }

    __host__ __device__
    inline const T& operator()(const vector_type& coords) const
    {
      return _data[offset(coords)];
    }

    __host__ __device__
    inline T& operator()(const vector_type& coords)
    {
      return _data[offset(coords)];
    }

    __host__
    inline void resize(const vector_type& sizes)
    {
      if (_sizes == sizes)
        return;

      CHECK_CUDA_RUNTIME_ERROR(cudaFree(_data));

      _sizes = sizes;
      _strides(N-1) = 1;
      for (int i = N - 2; i >= 0; --i)
        _strides(i) = _strides(i+1) * _sizes(i+1);

      size_t byte_size = sizeof(T) * size();
      CHECK_CUDA_RUNTIME_ERROR(cudaMalloc((void **)&_data, byte_size));
    }

    __host__
    inline void to_host_vector(std::vector<T>& host_vector) const
    {
      if (host_vector.size() != size())
        host_vector.resize(size());

      CHECK_CUDA_RUNTIME_ERROR(
        cudaMemcpy(host_vector.data(), _data, sizeof(T) * size(),
                   cudaMemcpyDeviceToHost));
    }

  protected:
    __host__ __device__
    inline size_t offset(const vector_type& coords) const
    {
      size_t index = 0;
      for (int i = 0; i < N; ++i)
        index += coords(i) * _strides(i);
      return index;
    }

  private:
    T *_data = nullptr;
    vector_type _sizes = vector_type();
    vector_type _strides = vector_type();
  };


} /* namespace Device */
} /* namespace DO */