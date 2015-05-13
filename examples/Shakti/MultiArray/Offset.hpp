#pragma once

#include <cuda.h>

#include "Matrix.hpp"


namespace DO { namespace Shakti {

  //! @{
  //! \brief Return the ND-coordinates.
  template <int N>
  __device__
  inline Vector<int, N> coords()
  {
    return -1;
  }

  template <>
  __device__
  inline Vector2i coords<2>()
  {
    return {
      blockDim.x * blockIdx.x + threadIdx.x,
      blockDim.y * blockIdx.y + threadIdx.y
    };
  }

  template <>
  __device__
  inline Vector3i coords<3>()
  {
    return {
      blockDim.x * blockIdx.x + threadIdx.x,
      blockDim.y * blockIdx.y + threadIdx.y,
      blockDim.z * blockIdx.z + threadIdx.z
    };
  }
  //! @}

  //! @{
  //! \brief Return grid strides.
  template <int N>
  __device__
  inline Vector<int, N> grid_strides()
  {
    return Vector<int, N>();
  }

  template <>
  __device__
  inline Vector2i grid_strides<2>()
  {
    return { blockDim.y * gridDim.y, 1 };
  }

  template <>
  __device__
  inline Vector3i grid_strides<3>()
  {
    return {
      blockDim.y * gridDim.y * blockDim.z * gridDim.z,
      blockDim.z * gridDim.z,
      1
    };
  }
  //! @}

  //! @{
  //! \brief Return the 1D-offset of the corresponding ND-coordinates.
  template <int N>
  __device__
  inline int offset()
  {
    return -1;
  }

  template <>
  __device__
  inline int offset<2>()
  {
    return coords<2>().dot(grid_strides<2>());
  }

  template <>
  __device__
  inline int offset<3>()
  {
    return coords<3>().dot(grid_strides<3>());
  }
  //! @}

}
}