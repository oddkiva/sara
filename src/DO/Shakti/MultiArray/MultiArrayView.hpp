#pragma once

#include <Utilities/ErrorCheck.hpp>

#include "Matrix.hpp"
#include "Strides.hpp"


namespace DO { namespace Shakti {

  //! \brief ND-array view class.
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
    //! \brief STL-compatible interface.
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
    //! \brief Slice type.
    using slice_type = MultiArrayView<T, N-1, Strides>;
    using const_slice_type = const MultiArrayView<T, N-1, Strides>;
    //! @}

    //! @{
    //! \brief Vector type.
    using vector_type = Vector<int, N>;
    using slice_vector_type = Vector<int, N-1>;
    //! @}

  public: /* methods */
    //! \brief Default constructor.
    __host__ __device__
    inline MultiArrayView() = default;

    //! \brief Constructor that wraps plain data with its known sizes.
    __host__ __device__
    inline MultiArrayView(value_type *data, const vector_type& sizes)
      : _data(data)
      , _sizes(sizes)
      , _strides(strides_type::compute(sizes))
    {
    }

    //! \brief Return the size vector of the MultiArray object.
    __host__ __device__
    const vector_type& sizes() const
    {
      return _sizes;
    }

    //! \brief Return the number of elements in the internal data array.
    __host__ __device__
    size_type size() const
    {
      return compute_size(_sizes);
    }

    //! \brief Return the i-th sizes of the multi-array.
    __host__ __device__
    int size(int i) const
    {
      return _sizes(i);
    }

    //! \brief Return the number of rows.
    __host__ __device__
    int rows() const
    {
      return _sizes(0);
    }

    //! \brief Return the number of columns.
    __host__ __device__
    int cols() const
    {
      return _sizes(1);
    }

    //! \brief Return the depth size.
    __host__ __device__
    int depth() const
    {
      return _sizes(2);
    }

    //! \brief Return the stride vector of the MultiArray object.
    __host__ __device__
    inline const vector_type& strides() const
    {
      return _strides;
    }

    //! \brief Return the stride value along the i-th dimension.
    __host__ __device__
    inline int stride(int i) const
    {
      return _strides(i);
    }

    //! @{
    //! \brief Return the array pointer.
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

    //! \brief Return true/false whether the array is empty.
    __host__ __device__
      inline bool empty() const
    {
      return _data == nullptr;
    }

    //! @{
    //! \brief Return the reference to array element at given coordinates.
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

    //! @{
    //! \brief Return the i-th slice of the MultiArray object.
    __host__ __device__
    inline slice_type operator[](int i)
    {
      slice_vector_type sizes(_sizes.tail(N-1));
      T * data = _data + _strides[0] * i;
      return slice_type(data, sizes);
    }

    __host__ __device__
    inline const_slice_type operator[](int i) const
    {
      slice_vector_type slice_sizes(_sizes.tail(N-1));
      const T * data = _data + _strides[0] * i;
      return const_slice_type(data, slice_sizes);
    }
    //! @}

    //! \brief Copy the ND-array content to a std::vector object.
    __host__
    inline void to_std_vector(std::vector<T>& array) const
    {
      if (array.size() != size())
        array.resize(size());

      CHECK_CUDA_RUNTIME_ERROR(
        cudaMemcpy(array.data(), _data, sizeof(T) * size(),
                   cudaMemcpyDeviceToHost));
    }

  protected:
    //! \brief Compute the raw size needed to allocate the internal data.
    __host__ __device__
    inline size_type compute_size(const vector_type& sizes) const
    {
      size_t sz = 1;
      for (int i = 0; i < N; ++i)
        sz *= sizes(i);
      return sz;
    }

    //! \brief Compute the 1D index of the corresponding coordinates.
    __host__ __device__
    inline int offset(const vector_type& pos) const
    {
      return pos.dot(_strides);
    }

  protected: /* data members. */
    //! \brief Internal storage array.
    value_type *_data = nullptr;
    //! \brief Vector of size along each dimension.
    vector_type _sizes = vector_type();
    //! \brief Vector of stride for each dimension.
    vector_type _strides = vector_type();
  };

} /* namespace Shakti */
} /* namespace DO */