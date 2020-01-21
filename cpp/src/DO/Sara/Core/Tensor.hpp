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

  //! @addtogroup MultiArray
  //! @{

  //! @{
  //! @brief Tensor classes are simple aliases for MultiArray-based classes.
  template <typename T, int Dimension, int StorageOrder = ColMajor>
  using TensorView = MultiArrayView<T, Dimension, StorageOrder>;

  template <typename T, int N, int StorageOrder = ColMajor,
            template <typename> class Allocator = std::allocator>
  using Tensor = MultiArray<T, N, StorageOrder, Allocator>;
  //! @}


  //! @{
  //! @brief In this world everything, everything is **ROW-MAJOR** like in
  //! TensorFlow.
  template <typename T, int N>
  using TensorView_ = TensorView<T, N, RowMajor>;

  template <typename T, int N>
  using Tensor_ = Tensor<T, N, RowMajor>;
  //! @}


  //! @{
  //! @brief View a std::vector as a rank-1 tensor.
  template <typename T>
  inline auto tensor_view(const std::vector<T>& v)
  {
    using TensorView = TensorView_<T, 1>;
    return TensorView{const_cast<T*>(v.data()),
                      typename TensorView::vector_type{int(v.size())}};
  }

  template <typename T>
  inline auto tensor_view(std::vector<T>& v)
  {
    using TensorView = TensorView_<T, 1>;
    return TensorView{v.data(),
                      typename TensorView::vector_type{int(v.size())}};
  }
  //! @}


  //! @brief Reinterpret the tensor view object  as an image view object.
  template <typename T, int N>
  auto image_view(TensorView_<T, N> in) -> ImageView<T, N>
  {
    auto out_sizes = in.sizes();
    std::reverse(out_sizes.data(), out_sizes.data() + N);
    return ImageView<T, N>{in.data(), out_sizes};
  }


  //! @{
  //! @brief Provide tensor views for generic ND-array objects.
  /*!
   * If a tensor coefficient is not a scalar but a matricial object, then the
   * function converts a ND-array of matrices into (N+2)D-array of scalars,
   * i.e.:
   *
   * ```
   * auto m = MultiArray<Matrix2f, 2>{2, 2};
   * m.flat_array().fill(Matrix2f::Identity());
   *
   * auto t = tensor_view(m);
   *
   * std::cout << "t(c,r,i,j) == m(i,j)(c,r) is "
   *           << std::boolapha << t(c,r,i,j) == m(i,j)(c,r) << std::endl;
   * // true.
   *
   * ```
   */
  template <typename T, int M, int N, int Dim, int StorageOrder>
  inline auto tensor_view(MultiArrayView<Array<T, M, N>, Dim, StorageOrder> in)
      -> MultiArrayView<T, Dim + 2, StorageOrder>
  {
    using tensor_type = TensorView<T, Dim + 2, StorageOrder>;
    using tensor_sizes_type = typename tensor_type::vector_type;

    const auto out_sizes =
        StorageOrder == RowMajor
            ? (tensor_sizes_type{} << in.sizes(), N, M).finished()
            : (tensor_sizes_type{} << M, N, in.sizes()).finished();

    return tensor_type{reinterpret_cast<T*>(in.data()), out_sizes};
  }

  template <typename T, int M, int N, int Dim, int StorageOrder>
  inline auto tensor_view(MultiArrayView<Matrix<T, M, N>, Dim, StorageOrder> in)
      -> MultiArrayView<T, Dim + 2, StorageOrder>
  {
    using tensor_type = TensorView<T, Dim + 2, StorageOrder>;
    using tensor_sizes_type = typename tensor_type::vector_type;

    const auto out_sizes =
        StorageOrder == RowMajor
            ? (tensor_sizes_type{} << in.sizes(), N, M).finished()
            : (tensor_sizes_type{} << M, N, in.sizes()).finished();

    return tensor_type{reinterpret_cast<T*>(in.data()), out_sizes};
  }
  //! @}


  //! @{
  //! @brief Provide tensor views for image objects.
  template <typename T, int N>
  inline auto tensor_view(ImageView<T, N> in) -> TensorView<T, N, RowMajor>
  {
    auto out_sizes = in.sizes();
    std::reverse(out_sizes.data(), out_sizes.data() + N);
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

  template <typename T, int M, int N, int Dim>
  inline auto tensor_view(ImageView<Array<T, M, N>, Dim> in)
      -> TensorView<T, Dim + 2, RowMajor>
  {
    using tensor_type = TensorView<T, Dim + 2, RowMajor>;
    using tensor_sizes_type = typename tensor_type::vector_type;

    auto out_sizes = (tensor_sizes_type{} << in.sizes(), N, M).finished();
    std::reverse(out_sizes.data(), out_sizes.data() + Dim);

    return tensor_type{reinterpret_cast<T*>(in.data()), out_sizes};
  }

  template <typename T, int M, int N, int Dim>
  inline auto tensor_view(ImageView<Matrix<T, M, N>, Dim> in)
      -> TensorView<T, Dim + 2, RowMajor>
  {
    using tensor_type = TensorView<T, Dim + 2, RowMajor>;
    using tensor_sizes_type = typename tensor_type::vector_type;

    auto out_sizes = (tensor_sizes_type{} << in.sizes(), N, M).finished();
    std::reverse(out_sizes.data(), out_sizes.data() + Dim);

    return tensor_type{reinterpret_cast<T*>(in.data()), out_sizes};
  }
  //! @}


  //! @{
  //! @brief Convert image data structures to tensor data structures.
  template <typename T, int N>
  inline auto to_cwh_tensor(ImageView<T, N> in) -> TensorView<T, N, RowMajor>
  {
    return tensor_view(std::move(in));
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

  //! @}

} /* namespace Sara */
} /* namespace DO */
