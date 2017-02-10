// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <array>

#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  //! @{
  //! @brief Tensor classes are simple aliases for MultiArray-based classes.
  template <typename T, int Dimension, int StorageOrder = ColMajor>
  using TensorView = MultiArrayView<T, Dimension, StorageOrder>;

  template <typename T, int N, int StorageOrder = ColMajor,
            template <typename> class Allocator = std::allocator>
  using Tensor = MultiArray<T, N, StorageOrder, Allocator>;
  //! @}


  //! @{
  //! @brief Convert image data structures to tensor data structures.
  template <typename T, int N>
  inline auto to_cwh_tensor(ImageView<T, N> in) -> TensorView<T, N, RowMajor>
  {
    auto out_sizes = in.sizes();
    std::reverse(out_sizes.data(), out_sizes.data() + N);
    return TensorView<T, N, RowMajor>{in.data(), out_sizes};
  }

  template <typename ChannelType, typename ColorSpace, int Dim>
  inline auto to_cwh_tensor(ImageView<Pixel<ChannelType, ColorSpace>, Dim> in)
      -> Tensor<ChannelType, Dim + 1, RowMajor>
  {
    using tensor_type = Tensor<ChannelType, Dim + 1, RowMajor>;
    using tensor_sizes_type = typename tensor_type::vector_type;
    constexpr auto num_channels = ColorSpace::size;

    auto out_sizes =
        (tensor_sizes_type{} << num_channels, in.sizes()).finished();
    std::reverse(out_sizes.data() + 1, out_sizes.data() + Dim + 1);

    auto out = tensor_type{out_sizes};
    auto out_slice_ptrs =
        std::array<typename tensor_type::pointer, num_channels>{};

    for (auto c = 0; c < num_channels; ++c)
      out_slice_ptrs[c] = out[c].data();

    for (auto in_i = in.begin(); in_i != in.end(); ++in_i)
    {
      for (auto c = 0; c < num_channels; ++c)
      {
        *out_slice_ptrs[c] = (*in_i)[c];
        ++out_slice_ptrs[c];
      }
    }

    return out;
  }
  //! @}

  //! @{
  //! @brief Provide tensor views for image objects.
  template <typename T, int N>
  inline auto tensor_view(ImageView<T, N> in) ->TensorView<T, N, RowMajor>
  {
    auto out_sizes = in.sizes();
    std::reverse(out_sizes.data(), out_sizes.data() + out_sizes.size());
    return TensorView<T, N, RowMajor>{in.data(), out_sizes};
  }

  template <typename ChannelType, typename ColorSpace, int Dim>
  inline auto tensor_view(ImageView<Pixel<ChannelType, ColorSpace>, Dim> in)
      -> TensorView<ChannelType, Dim + 1, RowMajor>
  {
    using tensor_type = TensorView<ChannelType, Dim + 1, ColMajor>;
    using tensor_sizes_type = typename tensor_type::vector_type;
    constexpr auto num_channels = ColorSpace::size;

    auto out_sizes =
        (tensor_sizes_type{} << in.sizes(), num_channels).finished();
    std::reverse(out_sizes.data(), out_sizes.data() + Dim);

    return TensorView<ChannelType, Dim + 1, RowMajor>{
        reinterpret_cast<ChannelType*>(in.data()), out_sizes};
  }
  //! @}

} /* namespace Sara */
} /* namespace DO */
