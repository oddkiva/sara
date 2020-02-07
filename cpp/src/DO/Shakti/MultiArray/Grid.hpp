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

#ifndef DO_SHAKTI_MULTIARRAY_GRID_HPP
#define DO_SHAKTI_MULTIARRAY_GRID_HPP

#include <DO/Shakti/MultiArray/MultiArrayView.hpp>


namespace DO { namespace Shakti {

  inline dim3 default_block_size_2d()
  {
    return dim3{16, 16};
  }

  inline dim3 default_block_size_3d()
  {
    return dim3{8, 8, 8};
  }

  inline dim3 grid_size_2d(const Vector2i& sizes, const int padded_width,
                           dim3 block_size = default_block_size_2d())
  {
    return dim3{
        (padded_width + block_size.x - 1) / block_size.x,
        (sizes.y() + block_size.y - 1) / block_size.y,
    };
  }

  inline dim3 grid_size_3d(const Vector3i& sizes, const int padded_width,
                           dim3 block_size = default_block_size_3d())
  {
    return dim3{(padded_width + block_size.x - 1) / block_size.x,
                (sizes.y() + block_size.y - 1) / block_size.y,
                (sizes.z() + block_size.z - 1) / block_size.z};
  }

  template <typename T, int N, typename Strides>
  inline dim3 grid_size_2d(const MultiArrayView<T, N, Strides>& data,
                           dim3 block_size = default_block_size_2d())
  {
    return grid_size_2d(data.sizes(), data.padded_width(), block_size);
  }

  template <typename T, int N, typename Strides>
  inline dim3 grid_size_3d(const MultiArrayView<T, N, Strides>& data,
                           dim3 block_size = default_block_size_3d())
  {
    return grid_size_3d(data.sizes(), data.padded_width(), block_size);
  }

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_MULTIARRAY_GRID_HPP */
