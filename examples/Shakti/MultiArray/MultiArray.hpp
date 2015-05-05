#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>

#include "MultiArrayView.hpp"


namespace DO { namespace Shakti {

  //! \brief ND-array class.
  template <typename T, int N, typename Strides = RowMajorStrides>
  class MultiArray : public MultiArrayView<T, N, Strides>
  {
    using self_type = MultiArray;
    using base_type = MultiArrayView<T, N, Strides>;
    using strides_type = Strides;

    using base_type::_data;
    using base_type::_sizes;
    using base_type::_strides;

  public:
    using vector_type = typename base_type::vector_type;
    using slice_vector_type = typename base_type::slice_vector_type;

    using slice_type = typename base_type::slice_type;
    using const_slice_type = const MultiArray<T, N-1, Strides>;

  public:
    //! @{
    //! \brief Constructor.
    __host__
    inline MultiArray() = default;

    __host__
    inline MultiArray(const vector_type& sizes)
      : base_type()
    {
      resize(sizes);
    }

    __host__
    inline MultiArray(const self_type& other)
      : base_type()
    {
      resize(other._sizes);
      CHECK_CUDA_RUNTIME_ERROR(
        cudaMemcpy(_data, other._begin, other.size() * sizeof(T),
                   cudaMemcpyHostToDevice));
    }

    __host__ __device__
    inline MultiArray(self_type&& other)
      : MultiArray()
    {
      _data = other._data;
      other._data = nullptr;
      _sizes = other._sizes;
      _strides = other._strides;
    }
    //! @}

    //! \brief Destructor.
    __host__
    inline ~MultiArray()
    {
      CHECK_CUDA_RUNTIME_ERROR(cudaFree(_data));
    }

    //! Resize the multi-array.
    __host__
    inline void resize(const vector_type& sizes)
    {
      if (_sizes == sizes)
        return;

      CHECK_CUDA_RUNTIME_ERROR(cudaFree(_data));

      _sizes = sizes;
      _strides = strides_type::compute(sizes);

      size_t byte_size = sizeof(T) * this->base_type::size();
      CHECK_CUDA_RUNTIME_ERROR(cudaMalloc((void **)&_data, byte_size));
    }
  };

} /* namespace Shakti */
} /* namespace DO */
