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

#ifndef DO_SHAKTI_MULTIARRAY_OFFSET_HPP
#define DO_SHAKTI_MULTIARRAY_OFFSET_HPP

#include <DO/Shakti/MultiArray/Matrix.hpp>


namespace DO { namespace Shakti {

  //! @{
  //! @brief Return the ND-coordinates.
  template <int N>
  __device__
  inline Vector<int, N> coords()
  {
    return Vector<int, N>{};
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
  //! @brief Return grid strides.
  template <int N>
  __device__
  inline Vector<int, N> grid_strides()
  {
    return Vector<int, N>{};
  }

  template <>
  __device__
  inline Vector2i grid_strides<2>()
  {
    return { 1, blockDim.x * gridDim.x };
  }

  template <>
  __device__
  inline Vector3i grid_strides<3>()
  {
    return {
      1,
      blockDim.x * gridDim.x,
      blockDim.x * gridDim.x * blockDim.y * gridDim.y,
    };
  }
  //! @}

  //! @{
  //! @brief Return the index of the corresponding ND-coordinates.
  template <int N>
  __device__
  inline int offset()
  {
    return -1;
  }

  template <>
  __device__
  inline int offset<1>()
  {
    return blockDim.x * blockIdx.x + threadIdx.x;
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

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_MULTIARRAY_OFFSET_HPP */
